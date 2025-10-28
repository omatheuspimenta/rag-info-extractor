# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "chromadb",
#     "faiss-cpu",
#     "json5",
#     "langchain",
#     "langchain-community",
#     "ollama",
#     "pymupdf",
#     "regex",
#     "rich",
# ]
# ///
import argparse
import logging
import ollama
from rich.logging import RichHandler
from pathlib import Path
import regex as re
import json
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain.schema import Document
import json5

# Set up logging
FORMAT = "%(message)s"

logging.basicConfig(
    format=FORMAT,
    level="INFO",
    handlers=[RichHandler(show_time=False, show_path=False, markup=False)],
)

logtext = logging.getLogger("rich")
logtext.setLevel(20)

class TextPreprocessor:
    def __init__(self, model: str = "qwen2.5:14b"):
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
        self.model = model

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

    def extract_pdf_text(self, pdf_path: Path) -> tuple:
        """
        Extract text from a PDF file.

        Args:
            pdf_path (Path): Path to the PDF file.
        Returns:
            str: Extracted text.
        """
        try:
            # Read pdf file into bits
            # with open(pdf_path, "rb") as f:
            #     pdf_data = f.read()
            loader = PyMuPDFLoader(pdf_path)
            data = loader.load()
            for i, doc in enumerate(data):
                doc.metadata.update({
                    "source": str(pdf_path),
                    "page": i + 1,
                    "total_pages": len(data)
                })
            # Tries to split into chunks to avoid overloading the LLM
            # langchain approach tries to split into chunks (but near of the chunk size find a space to avoid breaking words)
            # here we use chunk size of 500 and overlap of 100
            # to have some context in the next chunk
            # we can play with these values if needed
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100,separators=["\n\n", "\n", ". ", " ", ""])
            chunks = text_splitter.split_documents(data)
            # Add chunk-specific metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata["chunk_id"] = i
            embeddings = OllamaEmbeddings(model=self.model)
            vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
            vectorstore.add_documents(documents=chunks)

            
            retriever = vectorstore.as_retriever(
                search_type="mmr",  # Maximum Marginal Relevance for diversity
                search_kwargs={
                    "k": 6,           # Get more chunks
                    "fetch_k": 20,    # Consider more candidates before MMR
                    "lambda_mult": 0.5  # Balance between relevance and diversity
                }
            )
            return text_splitter, vectorstore, retriever, chunks
        except Exception as e:
            logtext.error(f"Error extracting text from {pdf_path}: {e}")
            return None, None, None, None
        
    def extract_json_text(self, json_path: Path) -> tuple:
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

            doc = Document(
                page_content=full_text,
                metadata={"source": str(json_path), "title": title, "type": "json"}
            )
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100,separators=["\n\n", "\n", ". ", " ", ""])
            chunks = text_splitter.split_documents([doc])
            
            embeddings = OllamaEmbeddings(model=self.model)
            vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
            vectorstore.add_documents(documents=chunks)

            
            retriever = vectorstore.as_retriever(
                search_type="mmr",  # Maximum Marginal Relevance for diversity
                search_kwargs={
                    "k": 6,           # Get more chunks
                    "fetch_k": 20,    # Consider more candidates before MMR
                    "lambda_mult": 0.5  # Balance between relevance and diversity
                }
            )
            return text_splitter, vectorstore, retriever, chunks
            

        except Exception as e:
            logtext.error(f"Error extracting text from {json_path}: {e}")
            return None, None, None

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


class LLMExtractor:
    def __init__(self, model_name: str = "qwen2.5:14b"):
        self.model_name = model_name
        # self.confidence_calc = ConfidenceCalculator()
        self._ensure_model_available()

        # Define extraction prompts
        self.prompts = {
            "bioproject": self._create_bioproject_prompt(),
            "experimental_conditions": self._create_experimental_prompt(),
            "organism_info": self._create_organism_prompt(),
        }
        
        self.preprocessor = TextPreprocessor()

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





    ######################
    
    def extract_from_pdf_with_llm(self, pdf_path: Path, extraction_type: str) -> dict[str, object]:
        """Extract using all chunks if retrieval isn't good enough"""
        try:
            text_splitter, vectorstore, retriever, chunks = self.preprocessor.extract_pdf_text(pdf_path)
            
            if chunks is None:
                return {}
            
            # Combine ALL chunks
            full_text = "\n\n".join([chunk.page_content for chunk in chunks])
            print(f"Total extracted text length: {len(full_text)} characters")
            print(">>>>>>>>>>>")
            print(full_text)
            print("<<<<<<<<<<<\n\n")
            
            # If it fits in context, use it all
            # if len(full_text) < 8000:
            #     return self.extract_with_llm(full_text, extraction_type, text_len=8000)
            
            
            
            # Otherwise, try retrieval
            #>>>>>>>>>>>>>>..
            # query = self._get_query_for_extraction_type(extraction_type)
            # relevant_docs = retriever.invoke(query)
            # combined_text = "\n".join([doc.page_content for doc in relevant_docs])
            
            
            # query = self._get_query_for_extraction_type(extraction_type)
            query = 'bioproject'
            results_with_scores = vectorstore.similarity_search_with_score(query, k=10)

            print("\nSimilarity scores:")
            for doc, score in results_with_scores:
                print(f"Score: {score:.3f} - Preview: {doc.page_content[:100]}...")

            # Take top k chunks
            relevant_docs = [doc for doc, score in results_with_scores[:5]]
            combined_text = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            return self.extract_with_llm(combined_text, extraction_type)
            
        except Exception as e:
            logtext.error(f"Error: {e}")
            return {}
        
    def extract_from_json_with_llm(self, json_path: Path, extraction_type: str) -> dict[str, object]:
        """Extract using all chunks if retrieval isn't good enough"""
        try:
            text_splitter, vectorstore, retriever, chunks = self.preprocessor.extract_json_text(json_path)

            if chunks is None:
                return {}
            
            # Combine ALL chunks
            full_text = "\n\n".join([chunk.page_content for chunk in chunks])
            
            # If it fits in context, use it all
            if len(full_text) < 8000:
                return self.extract_with_llm(full_text, extraction_type, text_len=8000)
            
            query = self._get_query_for_extraction_type(extraction_type)
            results_with_scores = vectorstore.similarity_search_with_score(query, k=10)

            print("\nSimilarity scores:")
            for doc, score in results_with_scores:
                print(f"Score: {score:.3f} - Preview: {doc.page_content[:100]}...")

            # Take top k chunks
            relevant_docs = [doc for doc, score in results_with_scores[:5]]
            combined_text = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            return self.extract_with_llm(combined_text, extraction_type)
            
        except Exception as e:
            logtext.error(f"Error: {e}")
            return {}

    def _extract_simple(self, pdf_path: Path, extraction_type: str) -> dict[str, object]:
        """Fallback: simple extraction without RAG if vectorstore fails"""
        try:
            loader = PyMuPDFLoader(pdf_path)
            data = loader.load()
            
            # Combine all pages
            full_text = "\n\n".join([doc.page_content for doc in data])
            
            # Use your existing extract_with_llm with the full text
            # (limited by text_len parameter)
            return self.extract_with_llm(full_text, extraction_type, text_len=8000)
            
        except Exception as e:
            logtext.error(f"Fallback extraction failed for {pdf_path}: {e}")
            return {}
            
    def _get_query_for_extraction_type(self, extraction_type: str) -> str:
        """Generate targeted search queries for different extraction types"""
        queries = {
            "bioproject": "BioProject accession PRJNA PRJEB PRJDB database repository data availability supplementary material",
            "author": "author names affiliations corresponding author email contact institution department",
            "methods": "methods methodology experimental design protocol procedure materials technique",
            "results": "results findings data analysis statistical significant",
            "funding": "funding grant support acknowledgment financial agency NSF NIH",
            "species": "species organism strain sample bacterial fungal animal plant",
            "sequencing": "sequencing RNA-seq DNA-seq platform Illumina coverage depth reads",
            "software": "software tool package pipeline program analysis version",
            # Add more as you discover what you need
        }
        return queries.get(extraction_type, extraction_type)  # Fallback to extraction_type itself

    def _get_retrieval_count(self, extraction_type: str) -> int:
        """How many chunks to retrieve based on extraction type"""
        counts = {
            "bioproject": 3,  # Usually concentrated in one section
            "author": 2,      # Usually at the beginning
            "methods": 5,     # Can be spread across multiple sections
            "results": 6,     # Often extensive
            "funding": 2,     # Usually in acknowledgments
            "species": 4,     # Can appear throughout
        }
        return counts.get(extraction_type, 4)  # Default to 4
    
    ######################



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
                # format="json",
                # raw=True,
                options={
                    "temperature": 0.1,  # for consistency
                    "top_k": 10, # more focused
                    "top_p": 0.5,  # more conservative
                    "num_predict": 500,  # max tokens predict for response
                    "stream": False,
                },
            )
            print(">>>>>>>>",extraction_type,"\n")
            print(text)
            print(">>>>>>>>\n")
            raw_text = response.get("response") if isinstance(response, dict) else response.response
            # Parse the JSON response
            result = self._parse_llm_response(str(raw_text))
            print(f"LLM raw response for {extraction_type}:\n{raw_text}\nParsed result:\n{result}\n")
            # Calculate confidence score
            # for key, value in result.items():
            #     if isinstance(value, str) and value and value != "UNCERTAIN":
            #         confidence = self.confidence_calc.calculate_field_confidence(
            #             value, text, key
            #         )
            #         result[f"{key}_confidence"] = confidence
            #     elif isinstance(value, list):
            #         confidences = []
            #         for item in value:
            #             if isinstance(item, str) and item and item != "UNCERTAIN":
            #                 conf = self.confidence_calc.calculate_field_confidence(
            #                     item, text, key
            #                 )
            #                 confidences.append(conf)
            #         if confidences:
            #             result[f"{key}_confidence"] = sum(confidences) / len(
            #                 confidences
            #             )

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
        # self.confidence_calc = ConfidenceCalculator()

        # # WIP Processing statistics
        # self.stats = ProcessingStats()
        # self._processing_lock = threading.Lock()

        logtext.info(
            f"Initialized BiologicalDataExtractor with model {self.model_name} and confidence threshold {self.confidence_threshold}"
        )
        
    # def extract_from_file(self, file_path: Path) -> ArticleExtraction:
    #     """
        
    #     """
    #     start_time = datetime.now()
    #     try:
    #         if file_path.suffix.lower() == ".pdf":
    #             text = self.preprocessor.extract_pdf_text(file_path)
    #             title = file_path.stem # Use filename as title for PDFs
    #             # sections = self.preprocessor.identify_sections_frompdftxt(text)
    #         elif file_path.suffix.lower() == ".json":
    #             text, title = self.preprocessor.extract_json_text(file_path)
    #             # sections = self.preprocessor.identify_sections_fromjson(file_path)
    #         else:
    #             raise ValueError(f"Unsupported file format: {file_path.suffix}. Only PDF and JSON are supported.")
            
    #         if not text.strip():
    #             raise ValueError("Extracted text is empty.")
            
    #         # Extraction object
    #         extraction = ArticleExtraction(
    #             article_id=file_path.stem,
    #             title=title,
    #             processing_metadata = {
    #                 "file_path": str(file_path),
    #                 "processing_start": start_time.isoformat(),
    #                 "text_length": len(text)
    #             }
    #         )
            
    #         # Extract sections
    #         sections = self.preprocessor.identify_sections(text)
    #         extraction.processing_metadata['sections_found'] = list(sections.keys())
                       
    #         # Perform multi-pass extraction
    #         self._extract_bioproject_info(extraction, sections)
    #         self._extract_experimental_conditions(extraction, sections)
    #         self._extract_organism_info(extraction, sections)
    #         self._assess_supplementary_flags(extraction, text)
            
    #         # Calculate overall confidence summary
    #         self._calculate_confidence_summary(extraction)
            
    #         # Update processing metadata
    #         processing_time = (datetime.now() - start_time).total_seconds()
    #         extraction.processing_metadata.update({
    #             'processing_end': datetime.now().isoformat(),
    #             'processing_time_seconds': processing_time
    #         })
            
    #         # Update statistics
    #         with self._processing_lock:
    #             self.stats.processed_articles += 1
    #             if extraction.confidence_summary.get("overall", 0) >= self.confidence_threshold:
    #                 self.stats.high_confidence_extractions += 1
    #             else:
    #                 self.stats.uncertain_extractions += 1
    #         logtext.info(f"Successfully processed {file_path.name} in {processing_time:.2f} seconds.")
    #         return extraction
        
    #     except Exception as e:
    #         logtext.error(f"Failed to process {file_path.name}: {e}")
            
    #         # Create minimal extraction with error info
    #         extraction = ArticleExtraction(
    #             article_id=file_path.stem,
    #             title=file_path.name,
    #             processing_metadata={
    #                 'file_path': str(file_path),
    #                 'processing_start': start_time.isoformat(),
    #                 'processing_end': datetime.now().isoformat(),
    #                 'error': str(e)
    #             }
    #         )
            
    #         with self._processing_lock:
    #             self.stats.failed_extractions += 1
            
    #         return extraction


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
    logtext.info("Hello from llm_test.py!")
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


    # test if pdf and json extraction works
    # for file_path in input_files:
    #     logtext.info(f"Processing file: {file_path}")
    #     if file_path.suffix.lower() == '.pdf':
    #         text_splitter, vectorstore, retriever, chunks = extractor.preprocessor.extract_pdf_text(file_path)
    #         if not all([text_splitter, vectorstore, retriever]):
    #             logtext.error(f"Failed to process PDF: {file_path}")
    #             continue
            
    #         # Check text extraction quality
    #         if chunks and len(chunks) > 0:
    #             total_chars = sum(len(c.page_content.strip()) for c in chunks)
    #             sample_text = chunks[0].page_content[:1000].replace("\n", " ")
    #             logtext.info(f"Extracted {len(chunks)} chunks ({total_chars} characters) from {file_path}")
    #             logtext.info(f"Sample text: {sample_text}")
    #         else:
    #             logtext.error(f"No text extracted from PDF: {file_path}")    
                
                
    #     elif file_path.suffix.lower() == '.json':
    #         text_splitter, vectorstore, retriever, chunks = extractor.preprocessor.extract_json_text(file_path)
    #         if not all([text_splitter, vectorstore, retriever]):
    #             logtext.error(f"Failed to process JSON: {file_path}")
    #             continue
            
    #         # Check text extraction quality
    #         if chunks and len(chunks) > 0:
    #             total_chars = sum(len(c.page_content.strip()) for c in chunks)
    #             sample_text = chunks[0].page_content[:1000].replace("\n", " ")
    #             logtext.info(f"Extracted {len(chunks)} chunks ({total_chars} characters) from {file_path}")
    #             logtext.info(f"Sample text: {sample_text}")
    #         else:
    #             logtext.error(f"No text extracted from JSON: {file_path}")
    
    # test if LLM extraction works with pdf file
    for file_path in input_files:
        logtext.info(f"Processing file: {file_path}")
        if file_path.suffix.lower() == '.pdf':
            result = extractor.llm_extractor.extract_from_pdf_with_llm(file_path, extraction_type='bioproject')
            logtext.info(f"Extraction result for {file_path}:\n{result}")
        elif file_path.suffix.lower() == '.json':
            # For JSON files, extract full text first
            result = extractor.llm_extractor.extract_from_json_with_llm(file_path, extraction_type='bioproject')
            logtext.info(f"Extraction result for {file_path}:\n{result}")
            # if not text.strip():
            #     logtext.error(f"Extracted text is empty for JSON: {file_path}")
            #     continue
            # result = extractor.llm_extractor.extract_with_llm(text, extraction_type='bioproject')
            # logtext.info(f"Extraction result for {file_path}:\n{result}")
if __name__ == "__main__":
    main()
