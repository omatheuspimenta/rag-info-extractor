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

SYSTEM_PROMPT_GENOTYPE = """You are a scientific data extraction assistant specializing in plant genomics, particularly sugarcane and citrus research. Your task is to extract genotype information associated with NCBI accessions (BioProject, SRA, GEO, etc.) from scientific article text.

Key guidelines:
1. Focus on sugarcane and citrus organisms (including their microbiomes)
2. Extract ALL genotype information: strain names, cultivar names, wild-type, mutant descriptions, genetic backgrounds
3. Link each genotype to its corresponding NCBI accession (BioProject, SRA, GEO, etc.)
4. If information is uncertain or ambiguous, flag it and include the relevant text snippet
5. If no genotype or accession information is found, return empty results
6. Be precise - only extract what is explicitly stated in the text"""

class PromptTemplates:
    def __init__(self, text: str = ""):
        """
        
        """
        self.prompts = {
            'genotype': self._genotype_prompt(text),
            #TODO add more prompts here
        }

    def _genotype_prompt(self, article_full_text:str) -> str:
        """
        Creates the user prompt with the article text.
        
        Args:
            article_full_text (str): Full text of the scientific article
        
        Returns:
            str: Formatted prompt for the LLM
        """
        
        USER_PROMPT = f"""Extract genotype information and NCBI accessions from the following scientific article.

    Article text:
    {article_full_text}

    Return your response as JSON with this structure:
    {{
        "entries": [
            {{
                "accession": "BioProject ID or SRA ID or GEO ID",
                "accession_type": "BioProject|SRA|GEO|Other",
                "organism": "organism name (e.g., Saccharum officinarum, Citrus sinensis)",
                "genotype": "genotype/strain/cultivar description",
                "confidence": "high|medium|low",
                "evidence_text": "relevant excerpt from article (only if confidence is medium or low)"
            }}
        ],
        "notes": "any additional relevant notes or uncertainties"
    }}

    Rules:
    - If no genotype or accession found, return: {{"entries": [], "notes": "No genotype or accession information found"}}
    - Include evidence_text ONLY when confidence is "medium" or "low"
    - Set confidence to "high" only when genotype is explicitly stated with the accession
    - Set confidence to "medium" when genotype can be inferred from context
    - Set confidence to "low" when genotype is ambiguous or unclear
    - For microbiome studies, extract the host plant genotype if available"""

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
    # parser.add_argument('--confidence-threshold', type=float, default=0.7, help='Confidence threshold')
    # parser.add_argument('--max-workers', type=int, default=4, help='Maximum worker threads')
    # parser.add_argument('--file-limit', type=int, help='Limit number of files to process (for testing)')
    
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
