# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "ollama",
#     "pydantic",
#     "pymupdf",
#     "regex",
#     "rich",
# ]
# ///
import json
import argparse
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field, validator
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import ollama
import pymupdf as fitz
import regex as re
from rich.logging import RichHandler

# Set up logging
FORMAT = "%(message)s"

logging.basicConfig(
    format=FORMAT,
    level="INFO",
    handlers=[RichHandler(show_time=False, show_path=False, markup=False)],
)

logtext = logging.getLogger("rich")
logtext.setLevel(20)


@dataclass
class ProcessingStats:
    """Statistics for batch processing"""

    total_articles: int = 0
    processed_articles: int = 0
    high_confidence_extractions: int = 0
    uncertain_extractions: int = 0
    failed_extractions: int = 0
    processing_time: float = 0.0

class ExtractionResult(BaseModel):
    """Structured result for a single extraction field"""
    value: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    source_text: str = ""
    extraction_method: str = ""
    

class BioProjectInfo(BaseModel):
    """BioProject related information"""
    project_id: ExtractionResult = Field(default_factory=lambda: ExtractionResult(confidence=0.0))
    project_name: ExtractionResult = Field(default_factory=lambda: ExtractionResult(confidence=0.0))
    database: ExtractionResult = Field(default_factory=lambda: ExtractionResult(confidence=0.0))
    accession_numbers: list[ExtractionResult] = Field(default_factory=list)


class ExperimentalConditions(BaseModel):
    """Experimental conditions and methodology"""
    organism: ExtractionResult = Field(default_factory=lambda: ExtractionResult(confidence=0.0))
    species: ExtractionResult = Field(default_factory=lambda: ExtractionResult(confidence=0.0))
    strain_cultivar: ExtractionResult = Field(default_factory=lambda: ExtractionResult(confidence=0.0))
    treatment: list[ExtractionResult] = Field(default_factory=list)
    environment: ExtractionResult = Field(default_factory=lambda: ExtractionResult(confidence=0.0))
    temperature: ExtractionResult = Field(default_factory=lambda: ExtractionResult(confidence=0.0))
    precipitation: ExtractionResult = Field(default_factory=lambda: ExtractionResult(confidence=0.0))
    humidity: ExtractionResult = Field(default_factory=lambda: ExtractionResult(confidence=0.0))
    duration: ExtractionResult = Field(default_factory=lambda: ExtractionResult(confidence=0.0))
    sample_size: ExtractionResult = Field(default_factory=lambda: ExtractionResult(confidence=0.0))
    controls: list[ExtractionResult] = Field(default_factory=list)
    replicates: ExtractionResult = Field(default_factory=lambda: ExtractionResult(confidence=0.0))


class SupplementaryFlags(BaseModel):
    """Flags for potentially missing information"""
    likely_in_supplement: bool = False
    methods_referenced_elsewhere: bool = False
    incomplete_methodology: bool = False
    supplement_references: list[str] = Field(default_factory=list)


class ArticleExtraction(BaseModel):
    """Complete extraction result for one article"""
    article_id: str
    title: str = ""
    bioproject_info: BioProjectInfo = Field(default_factory=BioProjectInfo)
    experimental_conditions: ExperimentalConditions = Field(default_factory=ExperimentalConditions)
    supplementary_flags: SupplementaryFlags = Field(default_factory=SupplementaryFlags)
    processing_metadata: dict[str, Any] = Field(default_factory=dict)
    confidence_summary: dict[str, float] = Field(default_factory=dict)


class TextPreprocessor:
    def __init__(self):
        # TODO: include more sophisticated cleaning and normalization if needed
        self.section_patterns = {
            "methods": [
                r"materials?\s+and\s+methods?",
                r"methodology",
                r"experimental\s+(?:design|procedure|setup)",
                r"methods?",
                r"sample\s+(?:preparation|collection)",
                r"data\s+collection",
            ],
            "results": [r"results?", r"findings?", r"data\s+analysis"],
            "discussion": [r"discussion", r"conclusion"],
        }

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text

        Args:
            text (str): Raw text to clean.
        Returns:
            str: Cleaned text.
        """
        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text)
        # Remove non-printable characters except newlines
        text = re.sub(r"[^\x20-\x7E\n]", "", text)
        return text.strip()

    def _collect_paragraphs(self, section: dict[str, str]) -> list[str]:
        """
        Collect paragraphs from a section dictionary.

        Args:
            section (dict[str, str]): Section dictionary with paragraph keys.
        Returns:
            list[str]: List of paragraph texts.
        """
        paragraphs = []

        if "paragraphs" in section:
            paragraphs.extend(map(str, section["paragraphs"]))

        for sub in section.get("subsections", []):
            paragraphs.extend(self._collect_paragraphs(sub))

        return paragraphs

    def _collect_sections(self, sections: list[dict]) -> dict[str, str]:
        """
        Collect sections from a list of section dictionaries.

        Args:
            sections (list[dict]): List of section dictionaries.
        Returns:
            dict[str, str]: Dictionary with section names as keys and section texts as values.
        """
        result = {}
        for section in sections:
            title = section.get("title")
            if title:
                all_paragraphs = self._collect_paragraphs(section)
                result[title] = "".join(all_paragraphs)

        return result

    def extract_pdf_text(self, pdf_path: Path) -> str:
        """
        Extract text from a PDF file.

        Args:
            pdf_path (Path): Path to the PDF file.
        Returns:
            str: Extracted text.
        """
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return self._clean_text(text)
        except Exception as e:
            logtext.error(f"Error extracting text from {pdf_path}: {e}")
            return ""

    def extract_json_text(self, json_path: Path) -> tuple[str, str]:
        """
        Extract full text and title from a JSON file.

        Args:
            json_path (Path): Path to the JSON file.
        Returns:
            tuple[str, str]: Extracted full text and title.
        """
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Try common field names for full text - can be improved in the future
            full_text_fields = ["full_text", "fulltext", "text", "content", "body"]
            title_fields = ["title", "Title", "article_title"]

            full_text = ""
            title = ""

            for field in full_text_fields:
                if field in data and data[field]:
                    temp = data[field]
                    if isinstance(temp, dict):
                        for subfield in full_text_fields:
                            if subfield in temp and temp[subfield]:
                                full_text = str(temp[subfield])
                                break
                    else:
                        full_text = str(temp)
                    break

            for field in title_fields:
                if field in data and data[field]:
                    title = str(data[field])
                    break

            return self._clean_text(full_text), title

        except Exception as e:
            logtext.error(f"Error extracting text from {json_path}: {e}")
            return "", ""

    def identify_sections_fromjson(self, json_path: Path) -> dict[str, str]:
        """
        Identify and extract different sections of the paper from a JSON file.
        Args:
            json_path (Path): Path to the JSON file.
        Returns:
            dict[str, str]: Dictionary with section names as keys and section texts as values.
        """
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            sections = data.get("content", {}).get("sections", {})
            return self._collect_sections(sections)
        except Exception as e:
            logtext.error(f"Error identifying sections from {json_path}: {e}")
            return {}

    def identify_sections_frompdftxt(self, text: str) -> dict[str, str]:
        """Identify and extract different sections of the paper

        Args:
            text (str): Full text of the paper.
        Returns:
            dict[str, str]: Dictionary with section names as keys and section texts as values.
        """
        sections = {"full_text": text}
        text_lower = text.lower()

        for section_name, patterns in self.section_patterns.items():
            for pattern in patterns:
                matches = list(re.finditer(pattern, text_lower, re.IGNORECASE))
                if matches:
                    # Find the start of this section
                    start = matches[0].start()

                    # Find the start of the next major section
                    next_section_start = len(text)
                    for other_section, other_patterns in self.section_patterns.items():
                        if other_section != section_name:
                            for other_pattern in other_patterns:
                                other_matches = re.finditer(
                                    other_pattern,
                                    text_lower[start + 100 :],
                                    re.IGNORECASE,
                                )
                                for match in other_matches:
                                    next_section_start = min(
                                        next_section_start, start + 100 + match.start()
                                    )

                    section_text = text[start:next_section_start].strip()
                    if len(section_text) > 100:  # Only include substantial sections
                        sections[section_name] = section_text
                        break

        return sections

    def identify_sections(self, text: str) -> dict[str, str]:
        """Identify and extract different sections of the paper"""
        sections = {'full_text': text}
        text_lower = text.lower()
        
        for section_name, patterns in self.section_patterns.items():
            for pattern in patterns:
                matches = list(re.finditer(pattern, text_lower, re.IGNORECASE))
                if matches:
                    # Find the start of this section
                    start = matches[0].start()
                    
                    # Find the start of the next major section
                    next_section_start = len(text)
                    for other_section, other_patterns in self.section_patterns.items():
                        if other_section != section_name:
                            for other_pattern in other_patterns:
                                other_matches = re.finditer(other_pattern, text_lower[start+100:], re.IGNORECASE)
                                for match in other_matches:
                                    next_section_start = min(next_section_start, start + 100 + match.start())
                    
                    section_text = text[start:next_section_start].strip()
                    if len(section_text) > 100:  # Only include substantial sections
                        sections[section_name] = section_text
                        break
        
        return sections

class ConfidenceCalculator:
    
    def __init__(self):
        self.bioproject_patterns = [
            r'PRJNA\d+',
            r'bioproject\s*:?\s*([A-Z]+\d+)',
            r'project\s*id\s*:?\s*([A-Z]+\d+)',
            r'SRA\s*:?\s*([A-Z]+\d+)',
            r'GEO\s*:?\s*GSE\d+'
        ]
        
        self.supplement_indicators = [
            'see supplementary',
            'additional file',
            'supporting information',
            'supplemental',
            'detailed in supplement',
            'methods described elsewhere',
            'see methods',
            'data available'
        ]
    
    def _check_explicit_mention(self, value: str, source_text: str) -> float:
        """
        
        """
        value_lower = value.lower()
        source_lower = source_text.lower()
        
        if value_lower in source_lower:
            return 1.0
        
        # TODO: Improve using fuzzy matching if needed, e.g. Levenshtein distance or other NLP techniques
        # Partial matches
        words = value_lower.split()
        if len(words) > 1:
            matches = sum(1 for word in words if len(word) > 3 and word in source_lower)
            return matches / len(words)

        return 0.0

    def _assess_context_quality(self, value: str, source_text: str, field_type: str) -> float:
        """
        
        """
        # TODO: Implement more sophisticated context analysis if needed
        # For now, we check if the value appears in a relevant context
        context_indicators = {
            'organism': ['species', 'strain', 'cultivar', 'organism', 'plant', 'animal'],
            'treatment': ['treatment', 'condition', 'exposure', 'applied', 'subjected'],
            'temperature': ['temperature', '°c', 'degrees', 'celsius', 'fahrenheit', 'heated', 'cooled'],
            'environment': ['environment', 'condition', 'greenhouse', 'field', 'laboratory', 'growth chamber'],
            'bioproject': ['bioproject', 'project', 'database', 'repository', 'accession', 'submitted']
        }
        
        indicators = context_indicators.get(field_type, [])
        if not indicators:
            return 0.5  # Default for unknown field types
        
        source_lower = source_text.lower()
        indicator_matches = sum(1 for indicator in indicators if indicator in source_lower)
        
        return min(1.0, indicator_matches / len(indicators) * 2)

    def _check_pattern_match(self, value: str, field_type: str) -> float:
        """
        Check if value matches expected patterns for the field type
        """
        # TODO: Expand patterns for more field types as needed.
        patterns = {
            'bioproject': self.bioproject_patterns,
            'temperature': [r'\d+\.?\d*\s*[°]?[CF]', r'\d+\.?\d*\s*degrees?'],
            'organism': [r'[A-Z][a-z]+\s+[a-z]+'],  # Binomial nomenclature
            'duration': [r'\d+\s*(?:days?|weeks?|months?|hours?|minutes?)']
        }
        
        field_patterns = patterns.get(field_type, [])
        if not field_patterns:
            return 0.5
        
        for pattern in field_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                return 1.0
        
        return 0.0
    
    def _assess_completeness(self, value: str) -> float:
        """Assess completeness of extracted information"""
        if not value.strip():
            return 0.0
        
        # Basic completeness checks
        # TODO: Improve with more sophisticated checks if needed
        if len(value.strip()) < 3:
            return 0.3
        elif len(value.strip()) < 10:
            return 0.6
        else:
            return 1.0

    def calculate_field_confidence(self, value: str, source_text: str, field_type: str) -> float:
        """
        
        """
        if not value or not source_text:
            return 0.0
        
        confidence_factors = {
            'explicit_mention': self._check_explicit_mention(value, source_text),
            'context_quality': self._assess_context_quality(value, source_text, field_type),
            'pattern_match': self._check_pattern_match(value, field_type),
            'completeness': self._assess_completeness(value)
        }
        
        # Weighted average of confidence factors
        # can be tuned based on importance
        # TODO: Adjust weights based on empirical evaluation
        weights = {
            'explicit_mention': 0.4,
            'context_quality': 0.3,
            'pattern_match': 0.2,
            'completeness': 0.1
        }
        
        weighted_confidence = sum(
            confidence_factors[factor] * weights[factor] for factor in confidence_factors
        )
        
        return min(max(weighted_confidence, 0.0), 1.0)
        
class LLMExtractor:
    def __init__(self, model_name: str = "qwen2.5:14b"):
        self.model_name = model_name
        self.confidence_calc = ConfidenceCalculator()
        self._ensure_model_available()

        # Define extraction prompts
        self.prompts = {
            "bioproject": self._create_bioproject_prompt(),
            "experimental_conditions": self._create_experimental_prompt(),
            "organism_info": self._create_organism_prompt(),
        }

    def _ensure_model_available(self):
        """
        Ensure the specified model is available locally.
        """
        try:
            ollama.show(self.model_name)
            logtext.info(f"Model {self.model_name} is available locally.")
        except Exception as e:
            logtext.error(f"Model {self.model_name} is not available: {e}")
            logtext.info(
                f"Please run in terminal: ollama pull {self.model_name}\nEnsure you have the Ollama server running."
            )

    # Prompt templates
    # TODO: Refine prompts based on extraction needs
    # Can be expanded with more prompts for different extraction tasks
    # Can be improved with few-shot examples if needed
    def _create_bioproject_prompt(self) -> str:
        return """You are an expert at extracting biological database information from scientific papers.

Extract BioProject information from the following text. Look for:
- BioProject IDs (e.g., PRJNA123456)
- Project names or descriptions  
- Database references (SRA, GEO, etc.)
- Accession numbers

Example:
Input: "RNA-seq data has been deposited in NCBI under BioProject PRJNA123456. Additional data is available in GEO under accession GSE789."
Output:
{{
  "project_id": "PRJNA123456",
  "database": "NCBI",
  "accession_numbers": ["GSE789"]
}}

Rules:
- Return valid JSON only
- Use "UNCERTAIN" if information is unclear
- Leave fields empty if not found
- Be conservative with confidence

Text to analyze:
{text}

JSON Output:"""

    def _create_experimental_prompt(self) -> str:
        return """You are an expert at extracting experimental conditions from biological research papers.

Extract experimental conditions from the following text. Look for:
- Environmental conditions (temperature, humidity, precipitation)
- Treatment details
- Sample information
- Duration and timing
- Controls and replicates

Example:
Input: "Arabidopsis plants were grown at 22°C under long-day conditions (16h light/8h dark) in a growth chamber. Plants were treated with 100mM NaCl for 7 days. Three biological replicates were used."
Output:
{{
  "temperature": "22°C",
  "environment": "growth chamber, long-day conditions (16h light/8h dark)",
  "treatment": ["100mM NaCl"],
  "duration": "7 days",
  "replicates": "three biological replicates"
}}

Rules:
- Return valid JSON only
- Use "UNCERTAIN" if information is ambiguous
- Extract exact values when possible
- Include units when present

Text to analyze:
{text}

JSON Output:"""

    def _create_organism_prompt(self) -> str:
        return """You are an expert at extracting organism and species information from biological research papers.

Extract organism information from the following text. Look for:
- Species names (scientific and common names)
- Strains or cultivars
- Organism type or classification

Example:
Input: "Wild-type Arabidopsis thaliana (Columbia-0 ecotype) seeds were obtained from ABRC. Plants were grown under standard conditions."
Output:
{{
  "organism": "Arabidopsis thaliana",
  "strain_cultivar": "Columbia-0 ecotype",
  "species": "Arabidopsis thaliana"
}}

Rules:
- Return valid JSON only
- Use proper scientific nomenclature when possible
- Use "UNCERTAIN" if information is unclear
- Extract the most specific information available

Text to analyze:
{text}

JSON Output:"""

    def extract_with_llm(
        self, text: str, extraction_type: str, text_len: int = 4000
    ) -> dict[str, object]:
        """ """
        prompt = self.prompts[extraction_type].format(
            text=text[:text_len]
        )  # text length limit for ollama
        
        try:
            # TODO: Consider using chat sessions for maintaining context if needed
            # We're using the generate method due to only need the text response from the prompt
            # not the previous conversation history.
            # In the future, if we want to maintain context over multiple interactions,
            # we can switch to using a chat session.
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                format="json",
                raw=True,
                options={
                    "temperature": 0.1,  # for consistency
                    "top_k": 10, # more focused
                    "top_p": 0.5,  # more conservative
                    "num_predict": 500,  # max tokens predict for response
                    "stream": False,
                },
            )
            print(">>>>>>>>\n")
            print(text)
            print(">>>>>>>>\n")
            raw_text = response.get("response") if isinstance(response, dict) else response.response
            # Parse the JSON response
            result = self._parse_llm_response(str(raw_text))
            print(f"LLM raw response for {extraction_type}:\n{raw_text}\nParsed result:\n{result}\n")
            # Calculate confidence score
            for key, value in result.items():
                if isinstance(value, str) and value and value != "UNCERTAIN":
                    confidence = self.confidence_calc.calculate_field_confidence(
                        value, text, key
                    )
                    result[f"{key}_confidence"] = confidence
                elif isinstance(value, list):
                    confidences = []
                    for item in value:
                        if isinstance(item, str) and item and item != "UNCERTAIN":
                            conf = self.confidence_calc.calculate_field_confidence(
                                item, text, key
                            )
                            confidences.append(conf)
                    if confidences:
                        result[f"{key}_confidence"] = sum(confidences) / len(
                            confidences
                        )

            return result
        except Exception as e:
            logtext.error(f"LLM extraction failed for {extraction_type}: {e}")
            return {}

    def _parse_llm_response(self, response: str) -> dict[str, object]:
        """
        Parse LLM JSON response, handling potential formatting issues

        Args:
            response (str): Raw response from the LLM.
        Returns:
            dict[str, object]: Parsed JSON as a dictionary, or empty dict on failure.
        """
        try:
            # Extract the JSON substring (first {...} block)
            json_match = re.search(r"\{[\s\S]*\}", response)
            if not json_match:
                logtext.warning("No JSON object found in LLM response.")
                return {}

            json_str = json_match.group()
            original_json = json_str  # keep for debugging

            # --- Cleanup phase ---
            json_str = json_str.strip()

            # Remove control characters, newlines, and tabs
            json_str = re.sub(r"[\n\r\t]", "", json_str)

            # Fix common issues: trailing commas before } or ]
            json_str = re.sub(r",\s*([}\]])", r"\1", json_str)

            # Remove duplicate commas
            json_str = re.sub(r",\s*,", ",", json_str)

            # Fix missing quotes around keys (e.g., {project: "id"} → {"project": "id"})
            json_str = re.sub(r"([{,]\s*)([A-Za-z0-9_]+)(\s*:)", r'\1"\2"\3', json_str)

            # Ensure valid JSON quotes (replace single quotes with double)
            json_str = re.sub(r"'", '"', json_str)

            # Try to parse cleaned JSON
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                logtext.warning(f"First JSON parse failed: {e}. Trying more relaxed parsing...")

                # Attempt with JSON5-like forgiving parser (if available)
                try:
                    import json5
                    return json5.loads(json_str)
                except Exception:
                    pass

                # As last resort, attempt Python literal eval (dangerous in general, safe here)
                import ast
                try:
                    return ast.literal_eval(json_str)
                except Exception:
                    logtext.error("Literal eval also failed.")

                logtext.error(f"Failed JSON parse after cleanup. Raw: {original_json}")
                return {}

        except Exception as e:
            logtext.error(f"Unexpected error parsing LLM JSON: {e}")
            logtext.debug(f"Raw response: {response}")
            return {}


class BiologicalDataExtractor:
    def __init__(
        self, model_name: str = "qwen2.5:14b", confidence_threshold: float = 0.7
    ):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold

        # WIP Initialize componentes
        self.preprocessor = TextPreprocessor()
        self.llm_extractor = LLMExtractor(model_name=self.model_name)
        self.confidence_calc = ConfidenceCalculator()

        # WIP Processing statistics
        self.stats = ProcessingStats()
        self._processing_lock = threading.Lock()

        logtext.info(
            f"Initialized BiologicalDataExtractor with model {self.model_name} and confidence threshold {self.confidence_threshold}"
        )

    def extract_from_file(self, file_path: Path) -> ArticleExtraction:
        """
        
        """
        start_time = datetime.now()
        try:
            if file_path.suffix.lower() == ".pdf":
                text = self.preprocessor.extract_pdf_text(file_path)
                title = file_path.stem # Use filename as title for PDFs
                # sections = self.preprocessor.identify_sections_frompdftxt(text)
            elif file_path.suffix.lower() == ".json":
                text, title = self.preprocessor.extract_json_text(file_path)
                # sections = self.preprocessor.identify_sections_fromjson(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}. Only PDF and JSON are supported.")
            
            if not text.strip():
                raise ValueError("Extracted text is empty.")
            
            # Extraction object
            extraction = ArticleExtraction(
                article_id=file_path.stem,
                title=title,
                processing_metadata = {
                    "file_path": str(file_path),
                    "processing_start": start_time.isoformat(),
                    "text_length": len(text)
                }
            )
            
            # Extract sections
            sections = self.preprocessor.identify_sections(text)
            extraction.processing_metadata['sections_found'] = list(sections.keys())
                       
            # Perform multi-pass extraction
            self._extract_bioproject_info(extraction, sections)
            self._extract_experimental_conditions(extraction, sections)
            self._extract_organism_info(extraction, sections)
            self._assess_supplementary_flags(extraction, text)
            
            # Calculate overall confidence summary
            self._calculate_confidence_summary(extraction)
            
            # Update processing metadata
            processing_time = (datetime.now() - start_time).total_seconds()
            extraction.processing_metadata.update({
                'processing_end': datetime.now().isoformat(),
                'processing_time_seconds': processing_time
            })
            
            # Update statistics
            with self._processing_lock:
                self.stats.processed_articles += 1
                if extraction.confidence_summary.get("overall", 0) >= self.confidence_threshold:
                    self.stats.high_confidence_extractions += 1
                else:
                    self.stats.uncertain_extractions += 1
            logtext.info(f"Successfully processed {file_path.name} in {processing_time:.2f} seconds.")
            return extraction
        
        except Exception as e:
            logtext.error(f"Failed to process {file_path.name}: {e}")
            
            # Create minimal extraction with error info
            extraction = ArticleExtraction(
                article_id=file_path.stem,
                title=file_path.name,
                processing_metadata={
                    'file_path': str(file_path),
                    'processing_start': start_time.isoformat(),
                    'processing_end': datetime.now().isoformat(),
                    'error': str(e)
                }
            )
            
            with self._processing_lock:
                self.stats.failed_extractions += 1
            
            return extraction
        
    
    def _extract_bioproject_info(self, extraction: ArticleExtraction, sections: dict[str, str]) -> None:
        """
        
        """
        # print(">>>>>>>> BIOPROJECT EXTRACTION <<<<<<<<\n")
        # print(sections.keys())
        # print("\n>>>>>>>>\n")
        # for k, v in sections.items():
            # print(">>>>", k, "<<<<\n")
            # print(v, "\n...\n")
        text_to_search = sections.get('full_text', sections.get('methods', ''))
        # print(">>>>>>>> TEXT TO SEARCH <<<<<<<<\n")
        # print(text_to_search)
        
        
        llm_result = self.llm_extractor.extract_with_llm(text_to_search, 'bioproject', text_len=len(text_to_search))
        
        if 'project_id' in llm_result:
            extraction.bioproject_info.project_id = ExtractionResult(
                value=llm_result['project_id'] if llm_result['project_id'] != 'UNCERTAIN' else None,
                confidence=llm_result.get('project_id_confidence', 0.0),
                source_text=self._extract_context(text_to_search, llm_result['project_id']),
                extraction_method='llm_extraction'
            )
            
        if 'project_name' in llm_result:
            extraction.bioproject_info.project_name = ExtractionResult(
                value=llm_result['project_name'] if llm_result['project_name'] != 'UNCERTAIN' else None,
                confidence=llm_result.get('project_name_confidence', 0.0),
                source_text=self._extract_context(text_to_search, llm_result['project_name']),
                extraction_method='llm_extraction'
            )
        
        if 'database' in llm_result:
            extraction.bioproject_info.database = ExtractionResult(
                value=llm_result['database'] if llm_result['database'] != 'UNCERTAIN' else None,
                confidence=llm_result.get('database_confidence', 0.0),
                source_text=self._extract_context(text_to_search, llm_result['database']),
                extraction_method='llm_extraction'
            )
        
        if 'accession_numbers' in llm_result and isinstance(llm_result['accession_numbers'], list):
            for acc_num in llm_result['accession_numbers']:
                if acc_num and acc_num != 'UNCERTAIN':
                    extraction.bioproject_info.accession_numbers.append(
                        ExtractionResult(
                            value=acc_num,
                            confidence=llm_result.get('accession_numbers_confidence', 0.0),
                            source_text=self._extract_context(text_to_search, acc_num),
                            extraction_method='llm_extraction'
                        )
                    )
                    
    
    def _extract_experimental_conditions(self, extraction: ArticleExtraction, sections: dict[str, str]) -> None:
        """
        
        """
        # Prioritize methods section
        text_to_search = sections.get('methods', sections.get('full_text', ''))
        
        llm_result = self.llm_extractor.extract_with_llm(text_to_search, 'experimental_conditions', text_len=len(text_to_search))
        
        # Map LLM results to extraction fields
        # TODO: Expand mappings as needed
        field_mapping = {
            'environment': 'environment',
            'temperature': 'temperature', 
            'precipitation': 'precipitation',
            'humidity': 'humidity',
            'duration': 'duration',
            'sample_size': 'sample_size',
            'replicates': 'replicates'
        }
        
        for llm_field, extraction_field in field_mapping.items():
            if llm_field in llm_result and llm_result[llm_field]:
                if llm_result[llm_field] != 'UNCERTAIN':
                    setattr(extraction.experimental_conditions, extraction_field, 
                           ExtractionResult(
                               value=llm_result[llm_field],
                               confidence=llm_result.get(f'{llm_field}_confidence', 0.0),
                               source_text=self._extract_context(text_to_search, llm_result[llm_field]),
                               extraction_method='llm_extraction'
                           ))
        
        # Handle treatment as list
        if 'treatment' in llm_result and isinstance(llm_result['treatment'], list):
            for treatment in llm_result['treatment']:
                if treatment and treatment != 'UNCERTAIN':
                    extraction.experimental_conditions.treatment.append(
                        ExtractionResult(
                            value=treatment,
                            confidence=llm_result.get('treatment_confidence', 0.0),
                            source_text=self._extract_context(text_to_search, treatment),
                            extraction_method='llm_extraction'
                        )
                    )
        
        # Handle controls as list
        if 'controls' in llm_result and isinstance(llm_result['controls'], list):
            for control in llm_result['controls']:
                if control and control != 'UNCERTAIN':
                    extraction.experimental_conditions.controls.append(
                        ExtractionResult(
                            value=control,
                            confidence=llm_result.get('controls_confidence', 0.0),
                            source_text=self._extract_context(text_to_search, control),
                            extraction_method='llm_extraction'
                        )
                    )
                    
    
    def _extract_organism_info(self, extraction: ArticleExtraction, sections: dict[str, str]):
        """
        
        Extract organism information
        """
        text_to_search = sections.get('methods', sections.get('full_text', ''))
        
        llm_result = self.llm_extractor.extract_with_llm(text_to_search, 'organism_info', text_len=len(text_to_search))
        
        if 'organism' in llm_result and llm_result['organism'] != 'UNCERTAIN':
            extraction.experimental_conditions.organism = ExtractionResult(
                value=llm_result['organism'],
                confidence=llm_result.get('organism_confidence', 0.0),
                source_text=self._extract_context(text_to_search, llm_result['organism']),
                extraction_method='llm_extraction'
            )
        
        if 'species' in llm_result and llm_result['species'] != 'UNCERTAIN':
            extraction.experimental_conditions.species = ExtractionResult(
                value=llm_result['species'],
                confidence=llm_result.get('species_confidence', 0.0),
                source_text=self._extract_context(text_to_search, llm_result['species']),
                extraction_method='llm_extraction'
            )
        
        if 'strain_cultivar' in llm_result and llm_result['strain_cultivar'] != 'UNCERTAIN':
            extraction.experimental_conditions.strain_cultivar = ExtractionResult(
                value=llm_result['strain_cultivar'],
                confidence=llm_result.get('strain_cultivar_confidence', 0.0),
                source_text=self._extract_context(text_to_search, llm_result['strain_cultivar']),
                extraction_method='llm_extraction'
            )
    
    
    def _assess_supplementary_flags(self, extraction: ArticleExtraction, full_text: str):
        """
        Assess flags for supplementary information
        
        """
        text_lower = full_text.lower()
        
        # Check for supplement indicators
        supplement_refs = []
        for indicator in self.confidence_calc.supplement_indicators:
            if indicator in text_lower:
                extraction.supplementary_flags.likely_in_supplement = True
                supplement_refs.append(indicator)
        
        extraction.supplementary_flags.supplement_references = supplement_refs
        
        # Check for external method references
        external_patterns = [
            r'methods?\s+described\s+(?:in|elsewhere)',
            r'see\s+(?:ref|reference)',
            r'detailed\s+protocol',
            r'standard\s+procedure'
        ]
        
        for pattern in external_patterns:
            if re.search(pattern, text_lower):
                extraction.supplementary_flags.methods_referenced_elsewhere = True
                break
        
        # Assess methodology completeness
        self._assess_methodology_completeness(extraction)

    def _assess_methodology_completeness(self, extraction: ArticleExtraction):
        """
        Assess if methodology information is complete
        """
        
        # Check if key experimental fields are missing or have low confidence
        key_fields = [
            extraction.experimental_conditions.organism,
            extraction.experimental_conditions.environment,
            extraction.experimental_conditions.sample_size
        ]
        
        missing_count = 0
        low_confidence_count = 0
        
        for field in key_fields:
            if not field.value:
                missing_count += 1
            elif field.confidence < 0.5:
                low_confidence_count += 1
        
        # Flag as incomplete if too many fields are missing or uncertain
        if missing_count >= 2 or low_confidence_count >= 2:
            extraction.supplementary_flags.incomplete_methodology = True
    
    def _extract_context(self, text: str, value: str, context_window: int = 200) -> str:
        """
        Extract context around a value for manual review
        """
        if not value or not text:
            return ""
        
        # Find the position of the value in text
        value_pos = text.lower().find(value.lower())
        if value_pos == -1:
            return text[:context_window]  # Return beginning if not found
        
        start = max(0, value_pos - context_window // 2)
        end = min(len(text), value_pos + len(value) + context_window // 2)
        
        context = text[start:end]
        if start > 0:
            context = "..." + context
        if end < len(text):
            context = context + "..."
            
        return context
    
    def _calculate_confidence_summary(self, extraction: ArticleExtraction):
        """
        Calculate overall confidence metrics
        """
        all_confidences = []
        
        # Collect all confidence scores
        bioproject_confidences = [
            extraction.bioproject_info.project_id.confidence,
            extraction.bioproject_info.project_name.confidence,
            extraction.bioproject_info.database.confidence
        ]
        
        experimental_confidences = [
            extraction.experimental_conditions.organism.confidence,
            extraction.experimental_conditions.species.confidence,
            extraction.experimental_conditions.environment.confidence,
            extraction.experimental_conditions.temperature.confidence,
            extraction.experimental_conditions.sample_size.confidence,
            extraction.experimental_conditions.duration.confidence
        ]
        
        # Add treatment confidences
        treatment_confidences = [t.confidence for t in extraction.experimental_conditions.treatment]
        
        # Add accession number confidences
        accession_confidences = [a.confidence for a in extraction.bioproject_info.accession_numbers]
        
        all_confidences.extend(bioproject_confidences)
        all_confidences.extend(experimental_confidences)
        all_confidences.extend(treatment_confidences)
        all_confidences.extend(accession_confidences)
        
        # Filter out zero confidences (missing fields)
        non_zero_confidences = [c for c in all_confidences if c > 0]
        
        if non_zero_confidences:
            extraction.confidence_summary = {
                'overall': sum(non_zero_confidences) / len(non_zero_confidences),
                'bioproject_avg': sum(c for c in bioproject_confidences if c > 0) / max(1, len([c for c in bioproject_confidences if c > 0])),
                'experimental_avg': sum(c for c in experimental_confidences if c > 0) / max(1, len([c for c in experimental_confidences if c > 0])),
                'fields_extracted': len(non_zero_confidences),
                'high_confidence_fields': len([c for c in non_zero_confidences if c >= self.confidence_threshold])
            }
        else:
            extraction.confidence_summary = {
                'overall': 0.0,
                'bioproject_avg': 0.0,
                'experimental_avg': 0.0,
                'fields_extracted': 0,
                'high_confidence_fields': 0
            }
    
    def process_batch(self, file_paths: list[Path], max_workers: int = 4) -> list[ArticleExtraction]:
        """Process multiple files in parallel"""
        logtext.info(f"Starting batch processing of {len(file_paths)} files")
        
        start_time = datetime.now()
        results = []
        
        with self._processing_lock:
            self.stats.total_articles = len(file_paths)
            self.stats.processed_articles = 0
            self.stats.high_confidence_extractions = 0
            self.stats.uncertain_extractions = 0
            self.stats.failed_extractions = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_path = {
                executor.submit(self.extract_from_file, path): path 
                for path in file_paths
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Log progress
                    with self._processing_lock:
                        processed = self.stats.processed_articles + self.stats.failed_extractions
                        if processed % 10 == 0:
                            logtext.info(f"Progress: {processed}/{len(file_paths)} files processed")
                            
                except Exception as e:
                    logtext.error(f"Unexpected error processing {path}: {e}")
        
        # Calculate final statistics
        total_time = (datetime.now() - start_time).total_seconds()
        with self._processing_lock:
            self.stats.processing_time = total_time
        
        logtext.info(f"Batch processing completed in {total_time:.2f}s")
        self._log_processing_stats()
        
        return results
    
    
    def _log_processing_stats(self):
        """
        Log processing statistics
        """

        logtext.info("=== Processing Statistics ===")
        logtext.info(f"Total articles: {self.stats.total_articles}")
        logtext.info(f"Successfully processed: {self.stats.processed_articles}")
        logtext.info(f"High confidence extractions: {self.stats.high_confidence_extractions}")
        logtext.info(f"Uncertain extractions: {self.stats.uncertain_extractions}")
        logtext.info(f"Failed extractions: {self.stats.failed_extractions}")
        logtext.info(f"Success rate: {(self.stats.processed_articles / self.stats.total_articles * 100):.1f}%")
        logtext.info(f"High confidence rate: {(self.stats.high_confidence_extractions / max(1, self.stats.processed_articles) * 100):.1f}%")
        logtext.info(f"Total processing time: {self.stats.processing_time:.2f}s")
        logtext.info(f"Average time per article: {(self.stats.processing_time / max(1, self.stats.processed_articles)):.2f}s")
        
    def save_results(self, results: list[ArticleExtraction], output_path: Path):
        """
        Save extraction results to JSON file
        """
        output_data = {
            'extraction_metadata': {
                'timestamp': datetime.now().isoformat(),
                'model_used': self.model_name,
                'confidence_threshold': self.confidence_threshold,
                'statistics': asdict(self.stats)
            },
            'extractions': [extraction.model_dump() for extraction in results]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logtext.info(f"Results saved to {output_path}")
    
    def generate_review_report(self, results: list[ArticleExtraction], output_path: Path):
        """
        Generate a report of items requiring manual review
        """
        review_items = []
        
        for extraction in results:
            article_review = {
                'article_id': extraction.article_id,
                'title': extraction.title,
                'overall_confidence': extraction.confidence_summary.get('overall', 0.0),
                'requires_review': [],
                'supplementary_flags': extraction.supplementary_flags.model_dump()
            }
            
            # Check bioproject fields
            if extraction.bioproject_info.project_id.confidence < self.confidence_threshold:
                article_review['requires_review'].append({
                    'field': 'bioproject_id',
                    'value': extraction.bioproject_info.project_id.value,
                    'confidence': extraction.bioproject_info.project_id.confidence,
                    'context': extraction.bioproject_info.project_id.source_text
                })
            
            # Check experimental conditions
            experimental_fields = [
                ('organism', extraction.experimental_conditions.organism),
                ('species', extraction.experimental_conditions.species),
                ('environment', extraction.experimental_conditions.environment),
                ('temperature', extraction.experimental_conditions.temperature),
                ('sample_size', extraction.experimental_conditions.sample_size)
            ]
            
            for field_name, field_data in experimental_fields:
                if field_data.confidence < self.confidence_threshold and field_data.confidence > 0:
                    article_review['requires_review'].append({
                        'field': field_name,
                        'value': field_data.value,
                        'confidence': field_data.confidence,
                        'context': field_data.source_text
                    })
            
            # Only include articles that need review
            if article_review['requires_review'] or any(extraction.supplementary_flags.dict().values()):
                review_items.append(article_review)
        
        # Save review report
        review_report = {
            'review_metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_articles': len(results),
                'articles_needing_review': len(review_items),
                'confidence_threshold': self.confidence_threshold
            },
            'review_items': review_items
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(review_report, f, indent=2, ensure_ascii=False)
        
        logtext.info(f"Review report saved to {output_path}")
        logtext.info(f"Articles requiring manual review: {len(review_items)}/{len(results)}")
        

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Extract biological data from academic articles')
    parser.add_argument('--input', type=Path, help='Directory containing PDF and JSON files')
    parser.add_argument('--output', type=Path, help='Directory to save results')
    parser.add_argument('--model', default='qwen2.5:14b', help='Ollama model name. Available models can be listed with "ollama list" command. Actually we have deepseek-r1 and qwen2.5:14b')
    parser.add_argument('--confidence-threshold', type=float, default=0.7, help='Confidence threshold')
    parser.add_argument('--max-workers', type=int, default=4, help='Maximum worker threads')
    parser.add_argument('--file-limit', type=int, help='Limit number of files to process (for testing)')
    
    return parser
     
def main() -> None:
    logtext.info("Hello from get_info_llm.py!")

    parser = get_parser()
    args = parser.parse_args()
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Initialize extractor
    extractor = BiologicalDataExtractor(
        model_name=args.model,
        confidence_threshold=args.confidence_threshold
    )
    
    # Find input files (exclude non-PDF/JSON and 'failed_dois.json' file)
    input_files = []
    # Process each subdirectory in the input directory
    for folder in args.input.iterdir():
        if folder.is_dir():
            logtext.info(f"Processing folder: {folder.name}")
            for pattern in ['*.pdf', '*.json']:
                input_files.extend(folder.glob(pattern))
    
    # Filter out the 'failed_dois.json' file
    input_files = [f for f in input_files if f.name != 'failed_dois.json']

    
    if args.file_limit:
        input_files = input_files[:args.file_limit]
    
    logtext.info(f"Found {len(input_files)} files to process")

    if not input_files:
        logtext.warning("No input files found. Exiting.")
        return

    # Process files
    results = extractor.process_batch(input_files, max_workers=args.max_workers)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = args.output / f"extraction_results_{timestamp}.json"
    review_path = args.output / f"manual_review_{timestamp}.json"
    
    extractor.save_results(results, results_path)
    extractor.generate_review_report(results, review_path)

    logtext.info("Processing completed successfully!")

if __name__ == "__main__":
    main()
