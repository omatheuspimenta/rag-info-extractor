# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "docling",
#     "ollama",
#     "regex",
#     "rich",
# ]
# ///

import argparse
from pathlib import Path
import json
from pydoc import text
import regex as re
import ollama
import logging
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

from rich.logging import RichHandler
# Set up logging
FORMAT = "%(message)s"

logging.basicConfig(
    format=FORMAT,
    level="INFO",
    handlers=[RichHandler(show_time=False, show_path=False, markup=False)],
)

docling_logger = logging.getLogger('docling')
logtext = logging.getLogger("rich")
logtext.setLevel(20)

SYSTEM_PROMPT_GENOTYPE = """
You are a specialized scientific information extraction assistant with expertise in plant genomics, molecular biology, and bioinformatics. Your primary focus is extracting genotype-related information and NCBI accession identifiers from full-text research articles about plant species, particularly sugarcane (Saccharum spp.) and citrus (Citrus spp.).

Core Competencies:
- Deep understanding of plant taxonomy, cultivar naming conventions, and variety nomenclature
- Recognition of genotype information in various forms: cultivar names, variety designations, strain identifiers, line numbers, wild-type references, mutant designations, breeding codes, and genetic background descriptions
- Identification of NCBI database identifiers (BioProject, SRA, BioSample, GEO accessions)
- Ability to distinguish between genotype mentions and other biological concepts
- Recognition of implicit genotype information from experimental context

CRITICAL RULE: You MUST extract information ONLY from the provided article text. DO NOT use external knowledge, assumptions, or information from other sources. If information is not present in the article, explicitly indicate it as null or missing.
"""

class PromptTemplates:
    def __init__(self, text: str = ""):
        """
        
        """
        self.prompts = {
            'genotype': self._genotype_prompt(text),
            #TODO add more prompts here
        }

    def _genotype_prompt(self, article_full_text:str) -> str:    
    # Few-shot examples showing desired behavior
        FEW_SHOT_EXAMPLES = """
    Example 1 - Complete Information:

    Article excerpt: "RNA was extracted from leaf samples of sugarcane cultivar 'SP80-3280' and 'RB867515'. Libraries were sequenced and deposited in NCBI under BioProject PRJNA123456 with individual SRA accessions SRR9876543 and SRR9876544 for SP80-3280 and RB867515, respectively."

    Reasoning Steps:
    1. Identify genotypes: Two cultivars mentioned - 'SP80-3280' and 'RB867515'
    2. Identify accessions: BioProject PRJNA123456, SRA accessions SRR9876543 and SRR9876544
    3. Link genotypes to accessions: Text explicitly links SP80-3280 to SRR9876543 and RB867515 to SRR9876544
    4. Assess confidence: HIGH - explicit mentions with clear associations
    5. Determine source location: This is in main text (methods section)

    Output:
    {
        "entries": [
            {
                "accession": "SRR9876543",
                "accession_type": "SRA",
                "organism": "Saccharum spp.",
                "genotype": "SP80-3280",
                "confidence": "high",
                "source_location": "main text",
                "evidence_text": "SRA accessions SRR9876543 for SP80-3280"
            },
            {
                "accession": "SRR9876544",
                "accession_type": "SRA",
                "organism": "Saccharum spp.",
                "genotype": "RB867515",
                "confidence": "high",
                "source_location": "main text",
                "evidence_text": "SRR9876544 for RB867515"
            }
        ],
        "notes": "Both cultivars clearly identified with direct SRA accession links under BioProject PRJNA123456"
    }

    Example 2 - Missing Accession:

    Article excerpt: "We analyzed drought stress responses in citrus variety 'Pera' grown under field conditions. Detailed sequencing information is provided in Supplementary Table S3."

    Reasoning Steps:
    1. Identify genotypes: One variety mentioned - 'Pera'
    2. Identify accessions: No accessions in this excerpt
    3. Check supplementary reference: Article mentions Supplementary Table S3 contains sequencing info
    4. Assess confidence: MEDIUM - genotype is clear, but accession location is referenced
    5. Determine source location: Genotype in main text, accession in supplementary file

    Output:
    {
        "entries": [
            {
                "accession": null,
                "accession_type": null,
                "organism": "Citrus spp.",
                "genotype": "Pera",
                "confidence": "medium",
                "source_location": "main text",
                "evidence_text": "citrus variety 'Pera' grown under field conditions"
            }
        ],
        "notes": "Accession information referenced in Supplementary Table S3 but not provided in main text"
    }

    Example 3 - Table Format:

    Article excerpt: "Sequencing data for all samples are summarized in Table 2:

    Table 2. Sample information and sequencing accessions
    | Sample ID | Cultivar    | Tissue | SRA Accession | BioSample    |
    |-----------|-------------|--------|---------------|--------------|
    | S1        | Valencia    | Leaf   | SRR12345678   | SAMN10001234 |
    | S2        | Valencia    | Root   | SRR12345679   | SAMN10001235 |
    | S3        | Hamlin      | Leaf   | SRR12345680   | SAMN10001236 |

    All samples are deposited under BioProject PRJNA555555."

    Reasoning Steps:
    1. Identify table structure: This is clearly a table with pipe-delimited columns
    2. Parse table headers: Sample ID, Cultivar, Tissue, SRA Accession, BioSample
    3. Extract row data: Three samples with cultivars Valencia (2x) and Hamlin (1x)
    4. Link genotypes to accessions: Each row links cultivar to specific SRA and BioSample IDs
    5. Assess confidence: HIGH - explicit tabular mapping with clear associations
    6. Note BioProject: PRJNA555555 applies to all samples

    Output:
    {
        "entries": [
            {
                "accession": "SRR12345678",
                "accession_type": "SRA",
                "organism": "Citrus spp.",
                "genotype": "Valencia",
                "confidence": "high",
                "source_location": "main text",
                "evidence_text": "Table 2: Sample S1, Cultivar Valencia, SRA Accession SRR12345678"
            },
            {
                "accession": "SRR12345679",
                "accession_type": "SRA",
                "organism": "Citrus spp.",
                "genotype": "Valencia",
                "confidence": "high",
                "source_location": "main text",
                "evidence_text": "Table 2: Sample S2, Cultivar Valencia, SRA Accession SRR12345679"
            },
            {
                "accession": "SRR12345680",
                "accession_type": "SRA",
                "organism": "Citrus spp.",
                "genotype": "Hamlin",
                "confidence": "high",
                "source_location": "main text",
                "evidence_text": "Table 2: Sample S3, Cultivar Hamlin, SRA Accession SRR12345680"
            }
        ],
        "notes": "All samples under BioProject PRJNA555555. Information extracted from Table 2 with clear cultivar-to-accession mappings."
    }

    Example 4 - No Information Found:

    Article excerpt: "Our review summarizes recent advances in plant genomics and their applications in breeding programs."

    Reasoning Steps:
    1. Search for genotypes: No specific cultivars, varieties, or strains mentioned
    2. Search for accessions: No NCBI identifiers present
    3. Check context: This is a review article without experimental data
    4. Assess completeness: No extractable information

    Output:
    {
        "entries": [],
        "notes": "This appears to be a review article with no specific genotype information or NCBI accessions mentioned in the provided text"
    }
    """

        USER_PROMPT = f"""Extract all genotype and NCBI accession information from the scientific article below. You MUST follow these steps:

    STEP 1 - SYSTEMATIC READING:
    Read the ENTIRE article text carefully, section by section:
    - Abstract: Check for study overview and organism mentions
    - Introduction: Look for study context and varieties being investigated
    - Materials and Methods: PRIMARY source for genotypes and sample descriptions
    - Results: Check for any additional genotype details or data references
    - Discussion: May contain references to data deposition
    - Data Availability Statement: CRITICAL section for NCBI accessions
    - Supplementary Information references: Note what information is external
    - Acknowledgments: Sometimes contains data repository information
    - TABLES: Pay special attention to tables (often formatted with pipes |, dashes -, or aligned columns). Tables frequently contain:
        * Sample/accession mappings (e.g., "Sample ID | Cultivar | Accession")
        * Genotype lists with corresponding metadata
        * Sequencing run information with SRA/BioSample IDs
        * Look for column headers like: Sample, Cultivar, Variety, Genotype, Accession, BioProject, SRA, BioSample, Run ID

    STEP 2 - INFORMATION IDENTIFICATION:
    Target Information to Extract (ONLY from the provided text):
    a) Genotypes: cultivar names, variety names, breeding lines, strain identifiers, clone designations, wild-type references, mutant names, genetic backgrounds, germplasm accessions
    b) NCBI Accessions: BioProject IDs (PRJNA*), SRA IDs (SRR*, SRX*, SRP*), BioSample IDs (SAMN*, SAMD*, SAME*), GEO IDs (GSE*, GSM*), GenBank accessions

    SPECIAL ATTENTION TO TABLES:
    - Tables may appear as:
        * Pipe-delimited format: | Column1 | Column2 | Column3 |
        * Space-aligned columns with headers and data rows
        * CSV-like format with commas
        * Mixed format with irregular spacing
    - Look for table indicators: "Table X", "Table X.", numbered tables, or section headers like "Sample Information"
    - Common table column headers that contain target information:
        * Genotype columns: Cultivar, Variety, Genotype, Strain, Line, Clone, Accession (germplasm), Sample Name, Material
        * NCBI columns: SRA, BioSample, Run, Accession, GEO, GenBank, BioProject ID
    - Read tables row by row, extracting genotype-accession pairs from each row
    - If a BioProject is mentioned before/after the table, it applies to all entries in that table

    STEP 3 - ASSOCIATION ANALYSIS:
    - Link each genotype with its corresponding NCBI accession when both are present
    - If multiple genotypes share the same accession series, create separate entries
    - Note any ambiguities in associations

    STEP 4 - CONFIDENCE ASSESSMENT:
    - HIGH: Explicit mention with clear association between genotype and accession
    - MEDIUM: Genotype clear but accession referenced elsewhere, OR accession clear but genotype inferred from context
    - LOW: Ambiguous mentions, incomplete information, or indirect references

    STEP 5 - SOURCE TRACKING:
    - "main text": Information in the primary article body
    - "supplementary file": Referenced in supplementary materials (but actual content not in provided text)
    - "data availability statement": In data/code availability sections
    - "methods appendix": In detailed methods/materials sections
    - "not found": Information explicitly missing from the document

    STEP 6 - GENERATE OUTPUT:
    Create JSON following the exact format shown in the examples above.

    {FEW_SHOT_EXAMPLES}

    Now analyze this article:

    ==== ARTICLE TEXT BEGIN ====
    {article_full_text}
    ==== ARTICLE TEXT END ====

    CRITICAL REMINDERS:
    - Extract ONLY information present in the article text above
    - DO NOT use external knowledge or make assumptions
    - If information is missing, use null and explain in notes
    - Include evidence_text for medium/low confidence entries
    - Return ONLY valid JSON, no markdown, no code blocks, no additional text

    Required JSON Output Format:
    {{
        "entries": [
            {{
                "accession": "string or null",
                "accession_type": "BioProject | SRA | BioSample | GEO | GenBank | Other | null",
                "organism": "string (scientific name)",
                "genotype": "string (cultivar/variety/strain)",
                "confidence": "high | medium | low",
                "source_location": "main text | supplementary file | data availability statement | methods appendix | not found",
                "evidence_text": "string or null (required for medium/low confidence)"
            }}
        ],
        "notes": "string (observations, uncertainties, or explanations)"
    }}"""

        return USER_PROMPT


class ExtractText:
    def __init__(self):
        pass
    
    def pdf2str(self, pdf_path: Path) -> str:
        """
        
        """
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True
        pipeline_options.ocr_options.lang = ["es"]
        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=4, device=AcceleratorDevice.AUTO
        )

        doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        try:
            conv_result = doc_converter.convert(pdf_path)
            # return conv_result.document.export_to_text()
            return conv_result.document.export_to_markdown()
        except Exception as e:
            logtext.error(f"Error processing PDF file {pdf_path}: {e}")
            return ""
        
    def json2str(self, json_path: Path) -> tuple[str,str]:
        """
        
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
            
            return title, full_text
        except Exception as e:
            logtext.error(f"Error reading JSON file {json_path}: {e}")
            return "", ""
            
class ExtractInfo:
    def __init__(self, model: str = 'llama3.3:70b'):
        self.model = model
        self._check_model()
        # self.extractor = ExtractText()
        # self.prompter = PromptTemplates()
    
    def _check_model(self):
        try:
            ollama.show(self.model)
            logtext.info(f"Model {self.model} is available locally.")
        except Exception as e:
            logtext.error(f"Model {self.model} is not available: {e}")
            logtext.info(
                f"Please run in terminal: ollama pull {self.model}\nEnsure you have the Ollama server running."
            )

    def run_llm(self, text:str, objective: str) -> str:
        
        self.prompter = PromptTemplates(text)
        prompt = self.prompter.prompts.get(objective)
        # logtext.info(f"Using model: {self.model}")
        logtext.info("\n\n")
        logtext.info(f"Using prompt: {prompt}")
        if not prompt:
            logtext.error(f"No prompt found for objective: {objective}")
            return ""
        try:
            response = ollama.generate(
                model = self.model,
                prompt=prompt,
                system=SYSTEM_PROMPT_GENOTYPE,
                format="json",
                # raw=True,
                options={
                    "temperature": 0.1,  # for consistency
                    "top_k": 10, # more focused
                    "top_p": 0.5,  # more conservative
                    "num_predict": 2000,  # max tokens predict for response
                    "stream": False,
                    "num_ctx": 8192,  # context window
                },
            )   
            # get the class of response variable
            response_class = response.__class__.__name__
            logtext.info(f"Response class: {response_class}\n Response.response class: {response['response'].__class__.__name__}")
            
            raw_text = response["response"] if isinstance(response, dict) else response.response
            logtext.info(f"LLM response received.\n")
            logtext.info(raw_text)
            return raw_text
        except Exception as e:
            logtext.error(f"Error during LLM generation: {e}")
            return ""

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Extract biological data from academic articles')
    parser.add_argument('--input', type=Path, help='Directory containing PDF and JSON files')
    parser.add_argument('--output', type=Path, help='Directory to save results')
    parser.add_argument('--model', default='llama3.3:70b', help='Ollama model name. Available models can be listed with "ollama list" command. Actually we have llama3.1, llama3.3:70b and qwen2.5:14b')
    
    return parser

def main() -> None:
    print("Hello from llm_finder.py!")
    parser = get_parser()
    args = parser.parse_args()
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
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

    
    # if args.file_limit:
    #     input_files = input_files[:args.file_limit]
    
    logtext.info(f"Found {len(input_files)} files to process")

    if not input_files:
        logtext.warning("No input files found. Exiting.")
        return
    
    extractor = ExtractText()
    llm_extractor = ExtractInfo(model=args.model)
    for file_path in input_files:
        logtext.info(f"Processing file: {file_path}\n")
        if file_path.suffix.lower() == '.json':
            
            title, text = extractor.json2str(file_path)
            if text:
                logtext.info(f"Extracted text from {file_path.name} (Title: {title[:30]}...)")
                # logtext.info(f"Text snippet: {text}")
                logtext.info(f"Characters: {len(text)}")
                logtext.info(f"Words (approx): {len(text.split())}")
                logtext.info(f"Tokens (rough): {len(text.split()) * 1.3}")
                logtext.info(f"Running LLM extraction for Genotype data...")
                
                response = llm_extractor.run_llm(text=text, objective='genotype')
            else:
                logtext.warning(f"No text extracted from {file_path.name}")

        elif file_path.suffix.lower() == '.pdf':
            # extractor = ExtractText()
            text = extractor.pdf2str(file_path)
            if text:
                logtext.info(f"Extracted text from {file_path.name}")
                # logtext.info(f"Text snippet: {text}...")
                logtext.info(f"Characters: {len(text)}")
                logtext.info(f"Words (approx): {len(text.split())}")
                logtext.info(f"Tokens (rough): {len(text.split()) * 1.3}")
                logtext.info(f"Running LLM extraction for Genotype data...")
                # llm_extractor = ExtractInfo(model=args.model)
                response = llm_extractor.run_llm(text=text, objective='genotype')
            else:
                logtext.warning(f"No text extracted from {file_path.name}")


    # extractor = ExtractInfo(model=args.model)
if __name__ == "__main__":
    main()
