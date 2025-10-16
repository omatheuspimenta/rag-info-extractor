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

SYSTEM_PROMPT_GENOTYPE = """You are a specialized scientific information extraction assistant with expertise in plant genomics, molecular biology, and bioinformatics. Your primary focus is extracting genotype-related information and NCBI accession identifiers from full-text research articles about plant species, particularly sugarcane (Saccharum spp.) and citrus (Citrus spp.).

Core Competencies:
- Deep understanding of plant taxonomy, cultivar naming conventions, and variety nomenclature
- Recognition of genotype information in various forms: cultivar names, variety designations, strain identifiers, line numbers, wild-type references, mutant designations, breeding codes, and genetic background descriptions
- Identification of NCBI database identifiers (BioProject, SRA, BioSample, GEO accessions)
- Ability to distinguish between genotype mentions and other biological concepts
- Recognition of implicit genotype information from experimental context

Extraction Principles:
1. Thoroughness: Analyze the ENTIRE document, including abstract, introduction, methods, results, discussion, acknowledgments, data availability statements, and references to supplementary materials
2. Precision: Only extract information directly related to genotypes/cultivars used in the study
3. Context awareness: Understand that genotype information may be:
   - Explicitly stated: "cultivar 'Valencia'"
   - Implied from context: "commercial variety widely used in Brazil"
   - Referenced indirectly: "the same accessions as reported in Smith et al. 2020"
   - Distributed across sections: cultivar mentioned in methods, accession in data availability
4. Artifact tolerance: Handle noisy text from PDF extraction (irregular line breaks, OCR errors, formatting issues)
5. Transparency: When information is missing, ambiguous, or located elsewhere, explicitly document this

Output Requirements:
- Respond ONLY with valid JSON following the specified schema
- No markdown code blocks, no explanatory text outside the JSON structure
- Use the "notes" field for observations, uncertainties, or contextual information
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
            
        USER_PROMPT = f"""Extract all genotype and NCBI accession information from the following scientific article. Read the ENTIRE text carefully.
        Find in Materials and Methods, RNA-Seq analysis, Data Availability or Supplementary Information.

Target Information:
Genotypes: cultivar names, variety names, breeding lines, strain identifiers, clone designations, wild-type references, mutant names, genetic backgrounds, germplasm accessions, or any designation identifying specific plant genetic material.

NCBI Accessions: BioProject IDs (PRJNA*), SRA IDs (SRR*, SRX*, SRP*), BioSample IDs (SAMN*, SAMD*, SAME*), GEO IDs (GSE*, GSM*), GenBank accessions, or any NCBI database identifier.

Extraction Guidelines:

1. Association: Link each genotype with its corresponding NCBI accession when both are present. If multiple genotypes share the same accession, create separate entries.

2. Confidence Levels:
   - HIGH: Genotype and accession explicitly mentioned together OR genotype clearly stated with complete information
   - MEDIUM: Genotype inferred from context, or accession mentioned separately from genotype but linkable
   - LOW: Ambiguous mentions, incomplete information, or references to external sources

3. Source Location Tracking:
   - "main text" - information in the primary article body
   - "supplementary file" - referenced in supplementary materials/tables/datasets
   - "data availability statement" - mentioned in data/code availability sections
   - "methods appendix" - in methods or materials sections with detail
   - "not found" - information not provided anywhere in the document

4. Evidence Text: Include a brief quote (10-30 words) from the article for MEDIUM or LOW confidence entries to justify the extraction.

5. Missing Information Protocol:
   - If genotype is mentioned but accession is missing: create entry with accession as null, note where accession might be found
   - If accession is mentioned but genotype is unclear: create entry, use confidence "low" or "medium"
   - If information is explicitly stated to be in supplementary files: document this in source_location and notes
   - If neither is found: return empty entries array with explanatory note

6. Multiple Genotypes: Create separate entries for each distinct genotype, even if they share the same study or accession series.

 Article Text:
{article_full_text}

Required JSON Output Format:

{{
    "entries": [
        {{
            "accession": "string or null",
            "accession_type": "BioProject | SRA | BioSample | GEO | GenBank | Other | null",
            "organism": "string (scientific name)",
            "genotype": "string (cultivar/variety/strain name or description)",
            "confidence": "high | medium | low",
            "source_location": "main text | supplementary file | data availability statement | methods appendix | not found",
            "evidence_text": "string or null (required for medium/low confidence)"
        }}
    ],
    "notes": "string (observations, uncertainties, references to external sources, or explanations)"
}}

CRITICAL: Return ONLY the JSON object. No additional text, no markdown formatting, no code blocks. DO NOT CREATE information not present in the article, if the information is missing, return null or empty fields as specified.
        """

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
            return conv_result.document.export_to_text()
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
    def __init__(self, model: str = 'llama3.1'):
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
        # logtext.info(f"Using prompt: {prompt}")
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
                    "num_predict": 500,  # max tokens predict for response
                    "stream": False,
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
    parser.add_argument('--model', default='llama3.1', help='Ollama model name. Available models can be listed with "ollama list" command. Actually we have deepseek-r1 and qwen2.5:14b')
    
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
