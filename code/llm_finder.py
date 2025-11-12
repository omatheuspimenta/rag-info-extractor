# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "docling",
#     "ollama",
#     "regex",
#     "rich",
#     "semantic-text-splitter",
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
from semantic_text_splitter import MarkdownSplitter
from rich.logging import RichHandler
import multiprocessing as mp
# Set up logging
FORMAT = "%(message)s"

logging.basicConfig(
    format=FORMAT,
    level="INFO",
    handlers=[RichHandler(show_time=False, show_path=False, markup=False)],
)

docling_logger = logging.getLogger('docling')
logtext = logging.getLogger("rich")
logtext.setLevel(10)

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

class LLMResponseParser:
    """Parse LLM responses that contain <think> tags and extract the answer."""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean text by removing standalone line numbers and extra whitespace.
        
        Args:
            text: The text to clean
            
        Returns:
            Cleaned text
            
        Example:
            Input: "some text\\n\\n123\\n456\\nmore text"
            Output: "some text more text"
        """
        # Split into lines
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            stripped = line.strip()
            # Skip empty lines and lines that are only numbers
            if stripped and not stripped.isdigit():
                cleaned_lines.append(stripped)
        
        # Join with single space and collapse multiple spaces
        cleaned = ' '.join(cleaned_lines)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        return cleaned.strip()
    
    @staticmethod
    def extract_answer(response: str) -> str:
        """
        Extract the answer from LLM response, removing <think> blocks.
        
        Args:
            response: The full LLM response text
            
        Returns:
            The answer without the <think> block
            
        Example:
            Input: "<think>reasoning...</think>False"
            Output: "False"
        """
        # Remove <think>...</think> block and any whitespace
        cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        return cleaned.strip()
    
    @staticmethod
    def parse_boolean_answer(response: str) -> bool:
        """
        Parse a boolean answer from the LLM response.
        
        Args:
            response: The full LLM response text
            
        Returns:
            True or False based on the answer
            
        """
        answer = LLMResponseParser.extract_answer(response)
        answer_lower = answer.lower()
        
        if answer_lower in ('true', 'yes', '1'):
            return True
        else:
            return False

class PromptTemplates:
    def __init__(self, text: str = ""):
        """
        
        """
        self.prompts = {
            'genotype': self._genotype_prompt(text),
            #TODO add more prompts here
        }

    # def _genotype_prompt(self, article_full_text:str) -> str:    
    # # Few-shot examples showing desired behavior
    #     FEW_SHOT_EXAMPLES = """
    # Example 1 - Complete Information:

    # Article excerpt: "RNA was extracted from leaf samples of sugarcane cultivar 'SP80-3280' and 'RB867515'. Libraries were sequenced and deposited in NCBI under BioProject PRJNA123456 with individual SRA accessions SRR9876543 and SRR9876544 for SP80-3280 and RB867515, respectively."

    # Reasoning Steps:
    # 1. Identify genotypes: Two cultivars mentioned - 'SP80-3280' and 'RB867515'
    # 2. Identify accessions: BioProject PRJNA123456, SRA accessions SRR9876543 and SRR9876544
    # 3. Link genotypes to accessions: Text explicitly links SP80-3280 to SRR9876543 and RB867515 to SRR9876544
    # 4. Assess confidence: HIGH - explicit mentions with clear associations
    # 5. Determine source location: This is in main text (methods section)

    # Output:
    # {
    #     "entries": [
    #         {
    #             "accession": "SRR9876543",
    #             "accession_type": "SRA",
    #             "organism": "Saccharum spp.",
    #             "genotype": "SP80-3280",
    #             "confidence": "high",
    #             "source_location": "main text",
    #             "evidence_text": "SRA accessions SRR9876543 for SP80-3280"
    #         },
    #         {
    #             "accession": "SRR9876544",
    #             "accession_type": "SRA",
    #             "organism": "Saccharum spp.",
    #             "genotype": "RB867515",
    #             "confidence": "high",
    #             "source_location": "main text",
    #             "evidence_text": "SRR9876544 for RB867515"
    #         }
    #     ],
    #     "notes": "Both cultivars clearly identified with direct SRA accession links under BioProject PRJNA123456"
    # }

    # Example 2 - Missing Accession:

    # Article excerpt: "We analyzed drought stress responses in citrus variety 'Pera' grown under field conditions. Detailed sequencing information is provided in Supplementary Table S3."

    # Reasoning Steps:
    # 1. Identify genotypes: One variety mentioned - 'Pera'
    # 2. Identify accessions: No accessions in this excerpt
    # 3. Check supplementary reference: Article mentions Supplementary Table S3 contains sequencing info
    # 4. Assess confidence: MEDIUM - genotype is clear, but accession location is referenced
    # 5. Determine source location: Genotype in main text, accession in supplementary file

    # Output:
    # {
    #     "entries": [
    #         {
    #             "accession": null,
    #             "accession_type": null,
    #             "organism": "Citrus spp.",
    #             "genotype": "Pera",
    #             "confidence": "medium",
    #             "source_location": "main text",
    #             "evidence_text": "citrus variety 'Pera' grown under field conditions"
    #         }
    #     ],
    #     "notes": "Accession information referenced in Supplementary Table S3 but not provided in main text"
    # }

    # Example 3 - Table Format:

    # Article excerpt: "Sequencing data for all samples are summarized in Table 2:

    # Table 2. Sample information and sequencing accessions
    # | Sample ID | Cultivar    | Tissue | SRA Accession | BioSample    |
    # |-----------|-------------|--------|---------------|--------------|
    # | S1        | Valencia    | Leaf   | SRR12345678   | SAMN10001234 |
    # | S2        | Valencia    | Root   | SRR12345679   | SAMN10001235 |
    # | S3        | Hamlin      | Leaf   | SRR12345680   | SAMN10001236 |

    # All samples are deposited under BioProject PRJNA555555."

    # Reasoning Steps:
    # 1. Identify table structure: This is clearly a table with pipe-delimited columns
    # 2. Parse table headers: Sample ID, Cultivar, Tissue, SRA Accession, BioSample
    # 3. Extract row data: Three samples with cultivars Valencia (2x) and Hamlin (1x)
    # 4. Link genotypes to accessions: Each row links cultivar to specific SRA and BioSample IDs
    # 5. Assess confidence: HIGH - explicit tabular mapping with clear associations
    # 6. Note BioProject: PRJNA555555 applies to all samples

    # Output:
    # {
    #     "entries": [
    #         {
    #             "accession": "SRR12345678",
    #             "accession_type": "SRA",
    #             "organism": "Citrus spp.",
    #             "genotype": "Valencia",
    #             "confidence": "high",
    #             "source_location": "main text",
    #             "evidence_text": "Table 2: Sample S1, Cultivar Valencia, SRA Accession SRR12345678"
    #         },
    #         {
    #             "accession": "SRR12345679",
    #             "accession_type": "SRA",
    #             "organism": "Citrus spp.",
    #             "genotype": "Valencia",
    #             "confidence": "high",
    #             "source_location": "main text",
    #             "evidence_text": "Table 2: Sample S2, Cultivar Valencia, SRA Accession SRR12345679"
    #         },
    #         {
    #             "accession": "SRR12345680",
    #             "accession_type": "SRA",
    #             "organism": "Citrus spp.",
    #             "genotype": "Hamlin",
    #             "confidence": "high",
    #             "source_location": "main text",
    #             "evidence_text": "Table 2: Sample S3, Cultivar Hamlin, SRA Accession SRR12345680"
    #         }
    #     ],
    #     "notes": "All samples under BioProject PRJNA555555. Information extracted from Table 2 with clear cultivar-to-accession mappings."
    # }

    # Example 4 - No Information Found:

    # Article excerpt: "Our review summarizes recent advances in plant genomics and their applications in breeding programs."

    # Reasoning Steps:
    # 1. Search for genotypes: No specific cultivars, varieties, or strains mentioned
    # 2. Search for accessions: No NCBI identifiers present
    # 3. Check context: This is a review article without experimental data
    # 4. Assess completeness: No extractable information

    # Output:
    # {
    #     "entries": [],
    #     "notes": "This appears to be a review article with no specific genotype information or NCBI accessions mentioned in the provided text"
    # }
    # """

    #     USER_PROMPT = f"""Extract all genotype and NCBI accession information from the scientific article below. You MUST follow these steps:

    # STEP 1 - SYSTEMATIC READING:
    # Read the ENTIRE article text carefully, section by section:
    # - Abstract: Check for study overview and organism mentions
    # - Introduction: Look for study context and varieties being investigated
    # - Materials and Methods: PRIMARY source for genotypes and sample descriptions
    # - Results: Check for any additional genotype details or data references
    # - Discussion: May contain references to data deposition
    # - Data Availability Statement: CRITICAL section for NCBI accessions
    # - Supplementary Information references: Note what information is external
    # - Acknowledgments: Sometimes contains data repository information
    # - TABLES: Pay special attention to tables (often formatted with pipes |, dashes -, or aligned columns). Tables frequently contain:
    #     * Sample/accession mappings (e.g., "Sample ID | Cultivar | Accession")
    #     * Genotype lists with corresponding metadata
    #     * Sequencing run information with SRA/BioSample IDs
    #     * Look for column headers like: Sample, Cultivar, Variety, Genotype, Accession, BioProject, SRA, BioSample, Run ID

    # STEP 2 - INFORMATION IDENTIFICATION:
    # Target Information to Extract (ONLY from the provided text):
    # a) Genotypes: cultivar names, variety names, breeding lines, strain identifiers, clone designations, wild-type references, mutant names, genetic backgrounds, germplasm accessions
    # b) NCBI Accessions: BioProject IDs (PRJNA*), SRA IDs (SRR*, SRX*, SRP*), BioSample IDs (SAMN*, SAMD*, SAME*), GEO IDs (GSE*, GSM*), GenBank accessions

    # SPECIAL ATTENTION TO TABLES:
    # - Tables may appear as:
    #     * Pipe-delimited format: | Column1 | Column2 | Column3 |
    #     * Space-aligned columns with headers and data rows
    #     * CSV-like format with commas
    #     * Mixed format with irregular spacing
    # - Look for table indicators: "Table X", "Table X.", numbered tables, or section headers like "Sample Information"
    # - Common table column headers that contain target information:
    #     * Genotype columns: Cultivar, Variety, Genotype, Strain, Line, Clone, Accession (germplasm), Sample Name, Material
    #     * NCBI columns: SRA, BioSample, Run, Accession, GEO, GenBank, BioProject ID
    # - Read tables row by row, extracting genotype-accession pairs from each row
    # - If a BioProject is mentioned before/after the table, it applies to all entries in that table

    # STEP 3 - ASSOCIATION ANALYSIS:
    # - Link each genotype with its corresponding NCBI accession when both are present
    # - If multiple genotypes share the same accession series, create separate entries
    # - Note any ambiguities in associations

    # STEP 4 - CONFIDENCE ASSESSMENT:
    # - HIGH: Explicit mention with clear association between genotype and accession
    # - MEDIUM: Genotype clear but accession referenced elsewhere, OR accession clear but genotype inferred from context
    # - LOW: Ambiguous mentions, incomplete information, or indirect references

    # STEP 5 - SOURCE TRACKING:
    # - "main text": Information in the primary article body
    # - "supplementary file": Referenced in supplementary materials (but actual content not in provided text)
    # - "data availability statement": In data/code availability sections
    # - "methods appendix": In detailed methods/materials sections
    # - "not found": Information explicitly missing from the document

    # STEP 6 - GENERATE OUTPUT:
    # Create JSON following the exact format shown in the examples above.

    # {FEW_SHOT_EXAMPLES}

    # Now analyze this article:

    # ==== ARTICLE TEXT BEGIN ====
    # {article_full_text}
    # ==== ARTICLE TEXT END ====

    # CRITICAL REMINDERS:
    # - Extract ONLY information present in the article text above
    # - DO NOT use external knowledge or make assumptions
    # - If information is missing, use null and explain in notes
    # - Include evidence_text for medium/low confidence entries
    # - Return ONLY valid JSON, no markdown, no code blocks, no additional text

    # Required JSON Output Format:
    # {{
    #     "entries": [
    #         {{
    #             "accession": "string or null",
    #             "accession_type": "BioProject | SRA | BioSample | GEO | GenBank | Other | null",
    #             "organism": "string (scientific name)",
    #             "genotype": "string (cultivar/variety/strain)",
    #             "confidence": "high | medium | low",
    #             "source_location": "main text | supplementary file | data availability statement | methods appendix | not found",
    #             "evidence_text": "string or null (required for medium/low confidence)"
    #         }}
    #     ],
    #     "notes": "string (observations, uncertainties, or explanations)"
    # }}"""

    #     return USER_PROMPT
    # def _genotype_prompt(self, article_full_text:str) -> str:    
    #         # Few-shot examples showing desired behavior
    #         FEW_SHOT_EXAMPLES = """
    #         Example 1 - Complete Information:

    #         Article excerpt: "RNA was extracted from leaf samples of sugarcane cultivar 'SP80-3280' and 'RB867515'. Libraries were sequenced and deposited in NCBI under BioProject PRJNA123456 with individual SRA accessions SRR9876543 and SRR9876544 for SP80-3280 and RB867515, respectively."

    #         Reasoning Steps:
    #         1. Identify genotypes: Two cultivars mentioned - 'SP80-3280' and 'RB867515'
    #         2. Identify accessions: BioProject PRJNA123456, SRA accessions SRR9876543 and SRR9876544
    #         3. Link genotypes to accessions: Text explicitly links SP80-3280 to SRR9876543 and RB867515 to SRR9876544
    #         4. Assess confidence: HIGH - explicit mentions with clear associations
    #         5. Determine source location: This is in main text (methods section)

    #         Output:
    #         {
    #             "entries": [
    #                 {
    #                     "accession": "SRR9876543",
    #                     "accession_type": "SRA",
    #                     "organism": "Saccharum spp.",
    #                     "genotype": "SP80-3280",
    #                     "confidence": "high",
    #                     "source_location": "main text",
    #                     "evidence_text": "SRA accessions SRR9876543 for SP80-3280"
    #                 },
    #                 {
    #                     "accession": "SRR9876544",
    #                     "accession_type": "SRA",
    #                     "organism": "Saccharum spp.",
    #                     "genotype": "RB867515",
    #                     "confidence": "high",
    #                     "source_location": "main text",
    #                     "evidence_text": "SRR9876544 for RB867515"
    #                 }
    #             ],
    #             "notes": "Both cultivars clearly identified with direct SRA accession links under BioProject PRJNA123456"
    #         }

    #         Example 2 - Missing Accession:

    #         Article excerpt: "We analyzed drought stress responses in citrus variety 'Pera' grown under field conditions. Detailed sequencing information is provided in Supplementary Table S3."

    #         Reasoning Steps:
    #         1. Identify genotypes: One variety mentioned - 'Pera'
    #         2. Identify accessions: No accessions in this excerpt
    #         3. Check supplementary reference: Article mentions Supplementary Table S3 contains sequencing info
    #         4. Assess confidence: MEDIUM - genotype is clear, but accession location is referenced
    #         5. Determine source location: Genotype in main text, accession in supplementary file

    #         Output:
    #         {
    #             "entries": [
    #                 {
    #                     "accession": null,
    #                     "accession_type": null,
    #                     "organism": "Citrus spp.",
    #                     "genotype": "Pera",
    #                     "confidence": "medium",
    #                     "source_location": "main text",
    #                     "evidence_text": "citrus variety 'Pera' grown under field conditions"
    #                 }
    #             ],
    #             "notes": "Accession information referenced in Supplementary Table S3 but not provided in main text"
    #         }

    #         Example 3 - Table Format:

    #         Article excerpt: "Sequencing data for all samples are summarized in Table 2:

    #         Table 2. Sample information and sequencing accessions
    #         | Sample ID | Cultivar    | Tissue | SRA Accession | BioSample    |
    #         |-----------|-------------|--------|---------------|--------------|
    #         | S1        | Valencia    | Leaf   | SRR12345678   | SAMN10001234 |
    #         | S2        | Valencia    | Root   | SRR12345679   | SAMN10001235 |
    #         | S3        | Hamlin      | Leaf   | SRR12345680   | SAMN10001236 |

    #         All samples are deposited under BioProject PRJNA555555."

    #         Reasoning Steps:
    #         1. Identify table structure: This is clearly a table with pipe-delimited columns
    #         2. Parse table headers: Sample ID, Cultivar, Tissue, SRA Accession, BioSample
    #         3. Extract row data: Three samples with cultivars Valencia (2x) and Hamlin (1x)
    #         4. Link genotypes to accessions: Each row links cultivar to specific SRA and BioSample IDs
    #         5. Assess confidence: HIGH - explicit tabular mapping with clear associations
    #         6. Note BioProject: PRJNA555555 applies to all samples

    #         Output:
    #         {
    #             "entries": [
    #                 {
    #                     "accession": "SRR12345678",
    #                     "accession_type": "SRA",
    #                     "organism": "Citrus spp.",
    #                     "genotype": "Valencia",
    #                     "confidence": "high",
    #                     "source_location": "main text",
    #                     "evidence_text": "Table 2: Sample S1, Cultivar Valencia, SRA Accession SRR12345678"
    #                 },
    #                 {
    #                     "accession": "SRR12345679",
    #                     "accession_type": "SRA",
    #                     "organism": "Citrus spp.",
    #                     "genotype": "Valencia",
    #                     "confidence": "high",
    #                     "source_location": "main text",
    #                     "evidence_text": "Table 2: Sample S2, Cultivar Valencia, SRA Accession SRR12345679"
    #                 },
    #                 {
    #                     "accession": "SRR12345680",
    #                     "accession_type": "SRA",
    #                     "organism": "Citrus spp.",
    #                     "genotype": "Hamlin",
    #                     "confidence": "high",
    #                     "source_location": "main text",
    #                     "evidence_text": "Table 2: Sample S3, Cultivar Hamlin, SRA Accession SRR12345680"
    # GET /
    #                 }
    #             ],
    #             "notes": "All samples under BioProject PRJNA555555. Information extracted from Table 2 with clear cultivar-to-accession mappings."
    #         }

    #         Example 4 - No Information Found:

    #         Article excerpt: "Our review summarizes recent advances in plant genomics and their applications in breeding programs."

    #         Reasoning Steps:
    #         1. Search for genotypes: No specific cultivars, varieties, or strains mentioned
    #         2. Search for accessions: No NCBI identifiers present
    #         3. Check context: This is a review article without experimental data
    #         4. Assess completeness: No extractable information

    #         Output:
    #         {
    #             "entries": [],
    #             "notes": "This appears to be a review article with no specific genotype information or NCBI accessions mentioned in the provided text"
    #         }
    #         """

    #         USER_PROMPT = f"""Extract all genotype information from the scientific article below. You MUST follow these steps:

    #         STEP 1 - SYSTEMATIC READING:
    #         Read the ENTIRE chunk text carefully, chunk by chunk. If the chunks contains sections:
    #         - Abstract: Check for study overview and organism mentions
    #         - Introduction: Look for study context and varieties being investigated
    #         - Materials and Methods: PRIMARY source for genotypes and sample descriptions
    #         - Results: Check for any additional genotype details or data references
    #         - Discussion: May contain references to data deposition
    #         - Data Availability Statement: CRITICAL section for NCBI accessions (e.g., BioProject IDs) that may apply to **all** samples mentioned elsewhere.
    #         - Supplementary Information references: Note what information is external
    #         - Acknowledgments: Sometimes contains data repository information
    #         - TABLES: Pay special attention to tables (often formatted with pipes |, dashes -, or aligned columns). Tables frequently contain:
    #             * Sample/accession mappings (e.g., "Sample ID | Cultivar | Accession")
    #             * Genotype lists with corresponding metadata
    #             * Sequencing run information with SRA/BioSample IDs
    #             * Look for column headers like: Sample, Cultivar, Variety, Genotype, Accession, BioProject, SRA, BioSample, Run ID

    #         STEP 2 - INFORMATION IDENTIFICATION:
    #         Target Information to Extract (ONLY from the provided text):
    #         a) Genotypes: cultivar names, variety names, breeding lines, strain identifiers, clone designations, wild-type references, mutant names, genetic backgrounds, germplasm accessions
    #         b) NCBI Accessions: BioProject IDs (PRJNA*), SRA IDs (SRR*, SRX*, SRP*), BioSample IDs (SAMN*, SAMD*, SAME*), GEO IDs (GSE*, GSM*), GenBank accessions

    #         SPECIAL ATTENTION TO TABLES:
    #         - Tables may appear as:
    #             * Pipe-delimited format: | Column1 | Column2 | Column3 |
    #             * Space-aligned columns with headers and data rows
    #             * CSV-like format with commas
    #             * Mixed format with irregular spacing
    #         - Look for table indicators: "Table X", "Table X.", numbered tables, or section headers like "Sample Information"
    #         - Common table column headers that contain target information:
    #             * Genotype columns: Cultivar, Variety, Genotype, Strain, Line, Clone, Accession (germplasm), Sample Name, Material
    #             * NCBI columns: SRA, BioSample, Run, Accession, GEO, GenBank, BioProject ID
    #         - Read tables row by row, extracting genotype-accession pairs from each row

    #         STEP 3 - ASSOCIATION ANALYSIS:
    #         - **ASSOCIATION HIERARCHY (CRITICAL):**
    #             1.  **PRIORITY 1 (Direct Link):** Your primary goal is to find **direct links** where a genotype (e.g., 'SP80-3280') is explicitly associated with an accession (e.g., 'SRR9876543') in the same sentence, paragraph, or table row. These are **high** confidence. (See Example 1 & 3).
    #             2.  **PRIORITY 2 (Global Link):** If a genotype is mentioned (e.g., in Materials & Methods) but has no *direct* accession, check if a single "global" BioProject (e.g., 'PRJNA123456') is provided for the *entire study* (e.g., in the Data Availability Statement). If so, you MAY associate the genotype with that BioProject, but set confidence to 'medium' unless the link is clearly implied (e.g., "all sequencing data for the cultivars mentioned...").
    #             3.  **PRIORITY 3 (Genotype Only):** If a genotype is clearly mentioned but **no** NCBI accession (BioProject, SRA, etc.) can be associated with it anywhere in the text, you MUST still extract the genotype. Create an entry with the genotype, but set `accession` and `accession_type` to `null`. (See Example 2).
    #         - Link each genotype with its corresponding NCBI accession *based on the hierarchy above*.
    #         - If multiple genotypes share the same accession series, create separate entries.
    #         - Note any ambiguities in associations in the 'notes' field.
    #         - HANDLE EDGE CASES:
    #             * Multiple organisms: Create separate entries for each organism-genotype-accession combination
    #             * Replicates: If same genotype has multiple accessions (biological/technical replicates), create one entry per accession
    #             * Pooled samples: If multiple genotypes are pooled into one accession, create one entry and note pooling in evidence_text
    #             * Wild-type vs mutants: Treat as distinct genotypes (e.g., "Col-0" vs "Col-0 mutant X")
    #             * Genotype synonyms: If article uses multiple names for same genotype, choose the most specific/official name

    #         STEP 4 - CONFIDENCE ASSESSMENT:
    #         - HIGH: Explicit mention with clear association between genotype and accession (Priority 1).
    #         - MEDIUM: Genotype clear but accession referenced elsewhere (Priority 2 or Example 2), OR accession clear but genotype inferred from context.
    #         - LOW: Ambiguous mentions, incomplete information, or indirect references (e.g., a very weak Priority 2 link).

    #         STEP 5 - SOURCE TRACKING:
    #         - "main text": Information in the primary article body
    #         - "supplementary file": Referenced in supplementary materials (but actual content not in provided text)
    #         - "data availability statement": In data/code availability sections
    #         - "methods appendix": In detailed methods/materials sections
    #         - "not found": Information explicitly missing from the document

    #         STEP 6 - REASONING OUTPUT (MANDATORY):
    #         Before generating the final JSON, you MUST output your reasoning process in plain text using this format:
            
    #         === REASONING PROCESS ===
    #         1. Article Scan Summary: [Brief overview of what sections contain relevant information]
    #         2. Genotypes Found: [List all genotypes identified with their locations]
    #         3. Accessions Found: [List all NCBI accessions identified with their locations]
    #         4. Association Logic: [Explain how you linked genotypes to accessions **using the 3-priority hierarchy**]
    #         5. Ambiguities/Challenges: [Note any unclear cases or missing information]
    #         6. Confidence Rationale: [Explain confidence levels assigned]
    #         === END REASONING ===

    #         STEP 7 - GENERATE JSON OUTPUT:
    #         After your reasoning, create JSON following the exact format shown in the examples above.

    #         {FEW_SHOT_EXAMPLES}

    #         Now analyze this article:

    #         ==== ARTICLE TEXT BEGIN ====
    #         {article_full_text}
    #         ==== ARTICLE TEXT END ====

    #         CRITICAL REMINDERS:
    #         - Extract ONLY information present in the article text above
    #         - DO NOT use external knowledge or make assumptions
    #         - If information is missing, use null and explain in notes
    #         - Include evidence_text for medium/low confidence entries
    #         - ALWAYS output your reasoning process first (=== REASONING PROCESS === section), then the JSON
    #         - The reasoning section helps ensure accuracy and allows validation of your extraction logic
    #         - For tables: Process systematically row-by-row, don't skip entries
    #         - For ambiguous cases: State your interpretation in reasoning, then reflect uncertainty in confidence level
    #         - Validate JSON format: Ensure all braces, brackets, and commas are correct
    #         - NO markdown code blocks (no ```json), NO additional commentary - just reasoning section followed by raw JSON

    #         OUTPUT STRUCTURE:
    #         === REASONING PROCESS ===
    #         [Your step-by-step analysis here]
    #         === END REASONING ===

    #         [Raw JSON output here - no code blocks, no markdown]

    #         Required JSON Output Format:
    #         {{
    #             "entries": [
    #                 {{
    #                     "accession": "string or null",
    #                     "accession_type": "BioProject | SRA | BioSample | GEO | GenBank | Other | null",
    #                     "organism": "string (scientific name)",
    #                     "genotype": "string (cultivar/variety/strain)",
    #                     "confidence": "high | medium | low",
    #                     "source_location": "main text | supplementary file | data availability statement | methods appendix | not found",
    #                     "evidence_text": "string or null (required for medium/low confidence)"
    #                 }}
    #             ],
    #             "notes": "string (observations, uncertainties, or explanations)"
    #         }}"""

    #         return USER_PROMPT

    def _genotype_prompt(self, article_full_text: str) -> str:
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

        Example 2 - Genotypes WITHOUT Accessions:

        Article excerpt: "In this study, S. officinarum L. cultivar Guitang 42 (GT42) was employed as plant material. It is an excellent cultivar with high-yield, high-sugar, lodging-resistance and suitable for mechanized production."

        Reasoning Steps:
        1. Identify genotypes: One cultivar clearly mentioned - 'Guitang 42' (also referred to as GT42)
        2. Identify accessions: NONE found in the text
        3. Check for any BioProject: NONE found
        4. Decision: MUST STILL EXTRACT THE GENOTYPE - This is a Priority 3 case
        5. Assess confidence: HIGH - genotype is explicitly stated as the study material
        6. Determine source location: main text (methods section)

        Output:
        {
            "entries": [
                {
                    "accession": null,
                    "accession_type": null,
                    "organism": "Saccharum officinarum",
                    "genotype": "Guitang 42 (GT42)",
                    "confidence": "high",
                    "source_location": "main text",
                    "evidence_text": "S. officinarum L. cultivar Guitang 42 (GT42) was employed as plant material"
                }
            ],
            "notes": "Genotype clearly identified but no NCBI accessions provided in the text"
        }

        Example 3 - Multiple Genotypes WITHOUT Accessions:

        Article excerpt: "Three sugarcane varieties, ROC22 (V1), LC05-136 (V2), and YT93-159 (V3), were treated with ultrasound waves. Healthy stems were selected, cut into single bud segments for the experiment."

        Reasoning Steps:
        1. Identify genotypes: THREE varieties clearly mentioned - ROC22, LC05-136, and YT93-159
        2. Identify accessions: NONE found in this excerpt
        3. Check for BioProject: NONE found
        4. Decision: MUST EXTRACT ALL THREE GENOTYPES - This is Priority 3 case
        5. Assess confidence: HIGH - all three varieties explicitly named
        6. Determine source location: main text (materials section)

        Output:
        {
            "entries": [
                {
                    "accession": null,
                    "accession_type": null,
                    "organism": "Saccharum spp.",
                    "genotype": "ROC22",
                    "confidence": "high",
                    "source_location": "main text",
                    "evidence_text": "Three sugarcane varieties, ROC22 (V1)"
                },
                {
                    "accession": null,
                    "accession_type": null,
                    "organism": "Saccharum spp.",
                    "genotype": "LC05-136",
                    "confidence": "high",
                    "source_location": "main text",
                    "evidence_text": "LC05-136 (V2)"
                },
                {
                    "accession": null,
                    "accession_type": null,
                    "organism": "Saccharum spp.",
                    "genotype": "YT93-159",
                    "confidence": "high",
                    "source_location": "main text",
                    "evidence_text": "YT93-159 (V3)"
                }
            ],
            "notes": "Three varieties clearly identified but no NCBI accessions provided in the text"
        }

        Example 4 - Parent Cultivars in Cross:

        Article excerpt: "Two cultivars 'LA Purple' (2n = 8× = 80) and 'MOL5829' (2n = 8× = 80) were selected as representatives of the two species, respectively. Among 98 F1 crosses, 20 plants of F1 progeny were selected."

        Reasoning Steps:
        1. Identify genotypes: Two parent cultivars - 'LA Purple' and 'MOL5829'
        2. Identify accessions: NONE found
        3. Decision: EXTRACT BOTH PARENTAL GENOTYPES - Priority 3 case
        4. Assess confidence: HIGH - both cultivars explicitly named as study materials
        5. Source location: main text

        Output:
        {
            "entries": [
                {
                    "accession": null,
                    "accession_type": null,
                    "organism": "Saccharum officinarum",
                    "genotype": "LA Purple",
                    "confidence": "high",
                    "source_location": "main text",
                    "evidence_text": "cultivars 'LA Purple' (2n = 8× = 80)"
                },
                {
                    "accession": null,
                    "accession_type": null,
                    "organism": "Saccharum robustum",
                    "genotype": "MOL5829",
                    "confidence": "high",
                    "source_location": "main text",
                    "evidence_text": "MOL5829' (2n = 8× = 80)"
                }
            ],
            "notes": "Two parental cultivars identified for crossing experiment, no NCBI accessions provided"
        }

        Example 5 - Table Format:

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

        Example 6 - No Information Found:

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

        USER_PROMPT = f"""Extract all genotype information from the scientific article below. You MUST follow these steps:

        STEP 1 - SYSTEMATIC READING:
        Read the ENTIRE text carefully, section by section. If the text contains sections:
        - Abstract: Check for study overview and organism mentions
        - Introduction: Look for study context and varieties being investigated
        - Materials and Methods: PRIMARY source for genotypes and sample descriptions
        - Results: Check for any additional genotype details or data references
        - Discussion: May contain references to data deposition
        - Data Availability Statement: CRITICAL section for NCBI accessions (e.g., BioProject IDs) that may apply to **all** samples mentioned elsewhere
        - Supplementary Information references: Note what information is external
        - Acknowledgments: Sometimes contains data repository information
        - TABLES: Pay special attention to tables (often formatted with pipes |, dashes -, or aligned columns). Tables frequently contain:
            * Sample/accession mappings (e.g., "Sample ID | Cultivar | Accession")
            * Genotype lists with corresponding metadata
            * Sequencing run information with SRA/BioSample IDs
            * Look for column headers like: Sample, Cultivar, Variety, Genotype, Accession, BioProject, SRA, BioSample, Run ID

        STEP 2 - GENOTYPE IDENTIFICATION (CRITICAL - READ CAREFULLY):
        
        **PRIMARY MISSION: Find ALL genotypes mentioned in the text, regardless of whether accessions exist.**
        
        Target genotype patterns to extract:
        a) Cultivar names: 'SP80-3280', 'Valencia', 'Guitang 42', 'GT42', 'ROC22'
        b) Variety designations: 'LC05-136', 'YT93-159', 'Pera', 'Hamlin'
        c) Breeding lines: 'RB867515', breeding codes with numbers and dashes
        d) Strain identifiers: laboratory codes, accession numbers (germplasm, NOT NCBI)
        e) Clone designations: clone numbers or names
        f) Parent names in crosses: 'LA Purple', 'MOL5829', parent cultivar names
        g) Wild-type references: 'Col-0', WT designations
        h) Mutant names: mutant lines, transgenic lines
        i) Multiple varieties in lists: "varieties A, B, and C" or "V1, V2, V3" notations
        
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
        
        **GENOTYPE RECOGNITION RULES:**
        - Genotypes often appear after keywords: "cultivar", "variety", "strain", "line", "clone", "genotype", "material"
        - May be in quotes: 'GenotypeName' or "GenotypeName"
        - May have parenthetical abbreviations: "Full Name (Abbr)"
        - May be listed with codes: "GenotypeName (V1)", "Sample A (Line 123)"
        - May appear in enumerated lists: "three varieties, X, Y, and Z"
        - Parent materials: "crossed with", "parent 1 and parent 2", "selected from"
        
        **CRITICAL SCANNING REQUIREMENT:**
        - Create a running list of EVERY genotype mention as you read
        - When you encounter phrases like "three varieties" or "two cultivars", you MUST find and extract ALL of them
        - Do NOT stop after finding one genotype if the text indicates multiple exist
        - Verify your final count matches stated numbers (e.g., if text says "three varieties", you must extract 3)

        Target NCBI Accessions to extract (secondary to genotypes):
        b) BioProject IDs (PRJNA*), SRA IDs (SRR*, SRX*, SRP*), BioSample IDs (SAMN*, SAMD*, SAME*), GEO IDs (GSE*, GSM*), GenBank accessions

        STEP 3 - ASSOCIATION ANALYSIS:
        - **ASSOCIATION HIERARCHY (CRITICAL):**
            1.  **PRIORITY 1 (Direct Link):** Find direct links where a genotype is explicitly associated with an accession in the same sentence, paragraph, or table row. These are **high** confidence.
            2.  **PRIORITY 2 (Global Link):** If a genotype is mentioned but has no direct accession, check if a single "global" BioProject is provided for the entire study. If so, you MAY associate the genotype with that BioProject, but set confidence to 'medium'.
            3.  **PRIORITY 3 (Genotype Only):** 
                *** If a genotype is clearly mentioned but NO NCBI accession can be associated with it anywhere in the text, you MUST STILL EXTRACT THE GENOTYPE ***
                *** Create an entry with the genotype, but set `accession` and `accession_type` to `null` ***
                *** This is NOT a failure - this is a successful extraction of genotype-only information ***
                *** Set confidence to 'high' if the genotype is explicitly stated as study material ***
                *** MANY scientific articles mention genotypes without providing NCBI accessions - this is normal and expected ***
            - HANDLE EDGE CASES:
                    * Multiple organisms: Create separate entries for each organism-genotype-accession combination
                    * Replicates: If same genotype has multiple accessions (biological/technical replicates), create one entry per accession
                    * Pooled samples: If multiple genotypes are pooled into one accession, create one entry and note pooling in evidence_text
                    * Wild-type vs mutants: Treat as distinct genotypes (e.g., "Col-0" vs "Col-0 mutant X")
                    * Genotype synonyms: If article uses multiple names for same genotype, choose the most specific/official name

        **PRIORITY 3 IS MANDATORY:** If you find genotypes but no accessions, DO NOT return empty entries. Extract the genotypes with null accessions.

        STEP 4 - CONFIDENCE ASSESSMENT:
        - HIGH: Explicit mention of genotype, clear association (Priority 1), OR genotype explicitly stated as study material even without accession (Priority 3)
        - MEDIUM: Genotype clear but accession referenced elsewhere (Priority 2), OR accession clear but genotype inferred from context
        - LOW: Ambiguous mentions, incomplete information, or indirect references

        STEP 5 - SOURCE TRACKING:
        - "main text": Information in the primary article body
        - "supplementary file": Referenced in supplementary materials (but actual content not in provided text)
        - "data availability statement": In data/code availability sections
        - "methods appendix": In detailed methods/materials sections
        - "not found": Information explicitly missing from the document

        STEP 6 - REASONING OUTPUT (MANDATORY):
        Before generating the final JSON, you MUST output your reasoning process in plain text using this format:
        
        === REASONING PROCESS ===
        1. Article Scan Summary: [Brief overview of what sections contain relevant information]
        2. Genotypes Found: [List EVERY genotype identified with their locations - verify count matches any stated numbers]
        3. Accessions Found: [List all NCBI accessions identified with their locations, or state "NONE FOUND"]
        4. Association Logic: [Explain how you linked genotypes to accessions using the 3-priority hierarchy, OR explain that genotypes were found but no accessions exist]
        5. Ambiguities/Challenges: [Note any unclear cases or missing information]
        6. Confidence Rationale: [Explain confidence levels assigned]
        7. Final Verification: [Confirm you extracted ALL mentioned genotypes]
        === END REASONING ===

        STEP 7 - GENERATE JSON OUTPUT:
        After your reasoning, create JSON following the exact format shown in the examples above.

        {FEW_SHOT_EXAMPLES}

        Now analyze this article:

        ==== ARTICLE TEXT BEGIN ====
        {article_full_text}
        ==== ARTICLE TEXT END ====

        CRITICAL REMINDERS BEFORE YOU START:
        1. **GENOTYPES ARE YOUR PRIMARY TARGET** - Extract them even without accessions
        2. **EMPTY ENTRIES ONLY IF NO GENOTYPES EXIST** - If you find ANY genotypes, you must extract them
        3. **Count verification** - If text says "three varieties", you must extract 3. If it says "two cultivars", extract 2
        4. **Priority 3 cases are COMMON and CORRECT** - Many papers mention genotypes without NCBI data
        5. **Read the ENTIRE text systematically** - Don't stop after finding one genotype if multiple are mentioned
        6. **Null accessions are VALID** - Setting accession=null when no accession exists is the correct behavior
        7. **High confidence for explicit genotypes** - Even without accessions, if a genotype is clearly stated as study material, use "high" confidence
        
        - Extract ONLY information present in the article text above
        - DO NOT use external knowledge or make assumptions
        - If information is missing, use null and explain in notes
        - Include evidence_text for medium/low confidence entries
        - ALWAYS output your reasoning process first (=== REASONING PROCESS === section), then the JSON
        - The reasoning section helps ensure accuracy and allows validation of your extraction logic
        - For tables: Process systematically row-by-row, don't skip entries
        - For ambiguous cases: State your interpretation in reasoning, then reflect uncertainty in confidence level
        - Validate JSON format: Ensure all braces, brackets, and commas are correct
        - NO markdown code blocks (no ```json), NO additional commentary - just reasoning section followed by raw JSON
        
        WRONG APPROACH EXAMPLES (DO NOT DO THIS):
        - Article mentions "cultivar GT42" → You return: {{"entries": [], "notes": "No accessions found"}}
        - Article says "three varieties: A, B, C" → You return only variety A
        - Article mentions "LA Purple and MOL5829" → You return: {{"entries": [], "notes": "No NCBI accessions"}}
        
        CORRECT APPROACH EXAMPLES (DO THIS):
        - Article mentions "cultivar GT42" → You return: GT42 entry with null accession, high confidence
        - Article says "three varieties: A, B, C" → You return 3 entries (A, B, and C)
        - Article mentions "LA Purple and MOL5829" → You return 2 entries with null accessions

        OUTPUT STRUCTURE:
        === REASONING PROCESS ===
        [Your step-by-step analysis here]
        === END REASONING ===

        [Raw JSON output here - no code blocks, no markdown]

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
                    "evidence_text": "string or null (quote from text showing where genotype was mentioned)"
                }}
            ],
            "notes": "string (observations, uncertainties, or explanations)"
        }}"""

        return USER_PROMPT

    def _chunk_genotype_prompt(self, chunk_text: str) -> str:
        """
        Determine if the given text chunk explicitly contains genotype-related information 
        about sugarcane or citrus that is clearly presented as part of the study’s materials 
        (e.g., Materials and Methods, Plant Material, Experimental Setup) 
        and directly associated with a BioProject ID (PRJNA*).

        Return ONLY one word: True or False.
        """
        USER_PROMPT = f"""
    You are a strict binary classifier.

    Evaluate the following text and decide whether it explicitly includes **genotype-related information** 
    about **sugarcane or citrus** — such as cultivar names, variety names, breeding lines, clone identifiers, 
    strain designations, mutant types, or germplasm accessions — that appear in the context of describing 
    the **materials used in the study** (for example, within sections like *Materials and Methods*, 
    *Plant Material*, *Experimental Procedures*, or similar).


    If the following conditions are satisfied:
    1. The text explicitly discusses sugarcane or citrus genotypes as part of the materials used in the study.  

    → Answer **True**.

    If any of these are missing (e.g., unrelated species, or no clear materials/genotype description) → Answer **False**.

    Text to analyze:
    \"\"\"{chunk_text}\"\"\"  

    Answer with ONLY one word: True or False.
    """
        return USER_PROMPT
    
    def _chunk_genotype_tables_prompt(self, chunk_text: str) -> str:
        """
        Determine if the given text chunk, which is likely a table in markdown format,
        contains explicit genotype-related information (e.g., cultivar, variety)
        for sugarcane or citrus.

        Return ONLY one word: True or False.
        """
        USER_PROMPT = f"""
    You are a strict binary classifier. Your task is to analyze the following text chunk.
    The chunk is likely a table (e.g., in markdown format with `|` and `---` separators)
    or text immediately surrounding a table (like a title or caption).

    You must determine if this text **explicitly lists genotypes for sugarcane or citrus in a table format**.

    A "genotype" includes: cultivar, variety, breeding line, strain, clone, or a specific 
    sample identifier that refers to a plant (e.g., 'SP80-3280', 'Valencia', 'RB867515').

    **CRITERIA FOR 'True':**
    Answer **True** ONLY IF the text chunk meets ALL of the following conditions:
    1.  The text clearly shows a **table structure** (e.g., with `|` pipes, `---` dashes, or clearly aligned columns of data).
    2.  The table **headers** (e.g., 'Cultivar', 'Variety', 'Genotype', 'Sample ID', 'Line') OR the table's **data cells** (e.g., 'SP80-3280', 'Pera') strongly indicate a list of plant genotypes.
    3.  There is **context** in the chunk (in the table, its title, or nearby text) explicitly mentioning 'sugarcane', 'citrus', 'Saccharum', or 'Citrus'.

    **CRITERIA FOR 'False':**
    Answer **False** if ANY of these are true:
    * The text is just a regular paragraph and **not a table**.
    * It is a table, but it lists other things (e.g., PCR primers, gene IDs, chemical concentrations, statistical results).
    * It is a genotype table, but for a different organism (e.g., 'Arabidopsis', 'E. coli', 'Zea mays').
    * No genotype or organism information is present.

    Text to analyze:
    \"\"\"{chunk_text}\"\"\"

    Answer with ONLY one word: True or False.
    """
        return USER_PROMPT

class ExtractText:
    def __init__(self):
        self._section_patterns = {
            'materials': [
                r'materials?(?:\s+and\s+methods?)?',
                r'\bbiological\s+materials?\b',
                r'\breagents?(?:\s+and\s+materials?)?\b',
                r'\bchemicals?(?:\s+and\s+reagents?)?\b',
                r'\bsamples?(?:\s+and\s+data)?\b',
                r'\bspecimens?\b',
                r'\bstrains?(?:\s+and\s+plasmids?)?\b',
                r'\bplasmids?(?:\s+and\s+strains?)?\b',
                r'\bcell\s+lines?\b',
                r'\bbacterial\s+strains?\b',
                r'\borganisms?\b',
                r'\bdna\s+samples?\b',
                r'\brna\s+samples?\b',
                r'\btissue\s+samples?\b',
            ],
            'methods': [
                r'\bmethods?(?:\s+and\s+materials?)?\b',
                r'\bexperimental\s+(?:procedures?|design|methods?)\b',
                r'\bmethodology\b',
                r'\bprocedures?\b',
                r'\bprotocols?\b',
                r'\bsequencing\s+(?:methods?|protocols?|procedures?)\b',
                r'\b(?:dna|rna)\s+(?:extraction|isolation|purification)\b',
                r'\blibrary\s+(?:preparation|construction)\b',
                r'\bde\s+novo\s+assembly\b',
                r'\bgenome\s+assembly\b',
                r'\bgenome\s+annotation\b',
            ],
            'data': [
                r'\bdata\s+availability\b',
                r'\bdata\s+access(?:ion)?\b',
                r'\bdata\s+sharing\b',
                r'\bavailability\s+(?:of\s+)?(?:data|materials?)\b',
                r'\baccession\s+(?:numbers?|codes?)\b',
                r'\b(?:ncbi|ena|ddbj)\s+accession\b',
                r'\b(?:geo|sra|arrayexpress)\s+accession\b',
                r'\bgenbank\s+accession\b',
                r'\bbioproject\b',
                r'\bbiosample\b',
                r'\bsequence\s+read\s+archive\b',
                r'\bgenome\s+assembly\s+data\b',
                r'\bdeposited\s+at\b',
                r'european\s+nucleotide\s+archive',
                r'database\s+(?:accession|deposition)',
                r'public(?:ly)?\s+available\s+data',
                r'raw\s+data',
                r'\bgenomic\s+data\b',
                r'\btranscriptomic\s+data\b',
                r'\bcode\s+availability\b',
                r'\bsoftware\s+availability\b',
                r'\bgithub\s+repository\b',
                r'\bsupplementary\s+(?:data|materials?|information|files?|tables?)\b',
                r'\badditional\s+files?\b',
                r'\bsupporting\s+information\b',
                r'\bdatasets?\b',
                r'\bsource\s+data\b'
            ]
        }

    def _match_section_type(self, header: str, patterns: list) -> bool:
        """Check if header matches any pattern in the list."""
        header_lower = header.lower()
        
        # Remove leading numbers (e.g., "2.1 Plant materials" -> "Plant materials")
        header_clean = re.sub(r'^\d+(?:\.\d+)*\s*', '', header_lower).strip()
        
        for pattern in patterns:
            if re.search(pattern, header_clean):
                return True
        return False
    
    def _is_markdown(self, text: str) -> bool:
        """Detect if text is markdown by checking for markdown headers."""
        # Look for markdown headers (# Header, ## Header, etc.)
        markdown_headers = re.findall(r'^#{1,6}\s+.+$', text, re.MULTILINE)
        return len(markdown_headers) > 2  # If we find multiple headers, it's likely markdown
    
    def _find_markdown_sections(self, text: str) -> list:
        """Find sections in markdown text using # headers, with number-based hierarchy."""
        markers = []
        header_pattern = r'^(#{1,6})\s+(.+?)$'
        
        current_top_level_type = None
        current_top_level_num = None # e.g., "2"
        current_top_level_md_level = 0 # e.g., 2 for ##

        for match in re.finditer(header_pattern, text, re.MULTILINE):
            level = len(match.group(1))  # Number of # symbols
            header_text = match.group(2).strip()
            position = match.start()
            
            section_type = None # Default to None
            
            # Check for number prefix
            num_match = re.match(r'^(\d+(?:\.\d+)*)', header_text)
            header_num = num_match.group(1) if num_match else None
            
            # 1. Check for an explicit match (e.g., "2 Materials and methods")
            found_explicit_match = False
            for s_type, patterns in self._section_patterns.items():
                if self._match_section_type(header_text, patterns):
                    section_type = s_type
                    found_explicit_match = True
                    
                    # This is a new top-level section
                    current_top_level_type = s_type
                    current_top_level_md_level = level
                    if header_num:
                        current_top_level_num = header_num.split('.')[0]
                    else:
                        current_top_level_num = None # Reset for non-numbered sections
                    break
            
            # 2. Check for inherited match (e.g., "2.1" is part of "2")
            if not found_explicit_match and current_top_level_type:
                # Inherit ONLY if:
                # A) It's a numbered sub-section (e.g., "2.1" follows "2")
                if (header_num and current_top_level_num and 
                    header_num.startswith(current_top_level_num + '.')):
                    section_type = current_top_level_type
                
                # B) [REMOVED] We no longer inherit type for non-numbered, 
                #    deeper headers as it was too greedy.

            # 3. Reset if we hit a new, non-matching section
            if not found_explicit_match and section_type is None:
                 current_top_level_type = None
                 current_top_level_num = None
                 current_top_level_md_level = 0
            
            # ALWAYS append the marker, even if type is None
            markers.append({
                'position': position,
                'header': header_text,
                'type': section_type, # This will be None for "References" etc.
                'level': level,
                'is_markdown': True
            })
        
        return markers
    
    def _find_plaintext_sections(self, text: str) -> list:
        """Find sections in plain text (inline headers)."""
        markers = []
        
        # Create a combined pattern for all section types
        all_patterns = []
        for section_type, patterns in self._section_patterns.items():
            for pattern in patterns:
                all_patterns.append((pattern, section_type))
        
        # Find all matches in the text
        for pattern, section_type in all_patterns:
            # Look for the pattern with optional preceding punctuation/whitespace
            full_pattern = r'(?:^|\.|\n)\s*(' + pattern + r'\.?)'
            
            for match in re.finditer(full_pattern, text, re.IGNORECASE):
                start_pos = match.start()
                header_text = match.group(1).strip()
                
                markers.append({
                    'position': start_pos,
                    'header': header_text,
                    'type': section_type,
                    'is_markdown': False
                })
        
        # Sort by position and remove duplicates
        markers.sort(key=lambda x: x['position'])
        
        seen_positions = set()
        unique_markers = []
        for marker in markers:
            if marker['position'] not in seen_positions:
                unique_markers.append(marker)
                seen_positions.add(marker['position'])
        
        return unique_markers
    
    def _extract_sections(self, text: str, force_markdown: bool | None = None) -> dict:
        """
        Extract materials, methods, and data availability sections from text.
        Automatically detects if text is markdown or plain text.
        
        Args:
            text: Text content (plain text or markdown)
            force_markdown: If True, treat as markdown. If False, treat as plain text.
                          If None (default), auto-detect.
            
        Returns:
            Dictionary with extracted sections
        """
        # Auto-detect or use forced mode
        if force_markdown is None:
            is_markdown = self._is_markdown(text)
        else:
            is_markdown = force_markdown
        
        # Find section markers based on format
        if is_markdown:
            all_markers = self._find_markdown_sections(text) # Finds ALL headers
        else:
            all_markers = self._find_plaintext_sections(text) # Finds ONLY relevant headers
        
        if not all_markers:
            return {
                'materials_text': '',
                'methods_text': '',
                'data_text': '',
                'combined_text': '',
                'found_sections': [],
                'format': 'markdown' if is_markdown else 'plaintext'
            }

        sections = {'materials': [], 'methods': [], 'data': []}
        found_sections_list = []

        if is_markdown:
            # --- NEW LOGIC FOR MARKDOWN ---
            combined_sections = []
            current_section_content = []
            current_section_marker = None

            for i, marker in enumerate(all_markers):
                # Find end of this marker's content
                end_pos = len(text)
                if i + 1 < len(all_markers):
                    end_pos = all_markers[i+1]['position']
                
                content = text[marker['position']:end_pos]
                
                if marker['type'] is None:
                    # This is an unrecognized section (e.g., "References")
                    # Finalize the previous section if it was a relevant one
                    if current_section_marker:
                        combined_sections.append({
                            'marker': current_section_marker,
                            'content': "".join(current_section_content)
                        })
                    # Reset
                    current_section_marker = None
                    current_section_content = []
                    continue

                # This is a relevant marker (materials, methods, data)
                
                if current_section_marker is None:
                    # Start of a new relevant section
                    current_section_marker = marker
                    current_section_content = [content]
                
                elif marker['type'] == current_section_marker['type']:
                    # Continuation of the current relevant section
                    current_section_content.append(content)
                
                else: # marker['type'] is different from current_section_marker['type']
                    # This is a new *relevant* section. Finalize the old one.
                    if current_section_marker:
                        combined_sections.append({
                            'marker': current_section_marker,
                            'content': "".join(current_section_content)
                        })
                    # Start the new one
                    current_section_marker = marker
                    current_section_content = [content]

            # Add the very last section if it was a relevant one
            if current_section_marker:
                 combined_sections.append({
                    'marker': current_section_marker,
                    'content': "".join(current_section_content)
                })

            # Now, populate the sections dict from our combined_sections
            for sec in combined_sections:
                marker = sec['marker']
                content = sec['content'].strip()
                sections[marker['type']].append({
                    'header': marker['header'],
                    'content': content,
                    'full_text': content
                })
            
            found_sections_list = [
                {
                    'type': s['marker']['type'], 
                    'header': s['marker']['header'], 
                    'level': s['marker'].get('level', None)
                }
                for s in combined_sections
            ]

        else:
            # --- ORIGINAL LOGIC FOR PLAINTEXT (which works fine) ---
            for i, marker in enumerate(all_markers): # all_markers only has relevant ones
                start_pos = marker['position']
                end_pos = len(text)
                if i + 1 < len(all_markers):
                    end_pos = all_markers[i + 1]['position']
                
                content = text[start_pos:end_pos].strip()
                sections[marker['type']].append({
                    'header': marker['header'],
                    'content': content,
                    'full_text': content
                })
            
            found_sections_list = [
                {'type': m['type'], 'header': m['header'], 'level': m.get('level', None)}
                for m in all_markers
            ]

        # Combine sections by type
        results = {
            'materials_text': '\n\n'.join([s['full_text'] for s in sections['materials']]),
            'methods_text': '\n\n'.join([s['full_text'] for s in sections['methods']]),
            'data_text': '\n\n'.join([s['full_text'] for s in sections['data']]),
            'combined_text': '\n\n'.join([
                s['full_text'] 
                for section_list in sections.values() 
                for s in section_list
            ]),
            'found_sections': found_sections_list,
            'format': 'markdown' if is_markdown else 'plaintext'
        }
        
        return results
    
    def pdf2str(self, pdf_path: Path) -> str:
        """
        
        """
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True
        pipeline_options.ocr_options.lang = ["es"]
        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=mp.cpu_count() - 1, device=AcceleratorDevice.AUTO
        )

        doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        try:
            conv_result = doc_converter.convert(pdf_path)
            # return conv_result.document.export_to_text()
            text = conv_result.document.export_to_markdown()
            # Extract sections from the markdown text
            sections = self._extract_sections(conv_result.document.export_to_text())
            for k, v in sections.items():
                if k == 'materials_text' and v not in ('', '#'):
                    text += '\nMATERIALS\n' + v
                elif k == 'methods_text' and v not in ('', '#'):
                    text += '\nMETHODS\n' + v
                elif k == 'data_text' and v not in ('', '#'):
                    text += '\nDATA AVAILABILITY\n' + v
            for _, table in enumerate(conv_result.document.tables):
                text += "\n" + table.export_to_markdown() + "\n"
            return text
        except Exception as e:
            logtext.error(f"Error processing PDF file {pdf_path}: {e}")
            return ""
    
    def _get_table_from_pdf(self, pdf_path: Path) -> str:
        """
        
        """
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True
        pipeline_options.ocr_options.lang = ["es"]
        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=mp.cpu_count() - 1, device=AcceleratorDevice.AUTO
        )

        doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        tables_md = ''
        try:
            conv_result = doc_converter.convert(pdf_path)
            for _, table in enumerate(conv_result.document.tables):
                tables_md += table.export_to_markdown() + "\n"
        except Exception as e:
            logtext.error(f"Error processing PDF file {pdf_path}: {e}")
        return tables_md
    
    def pdf2split(self, pdf_path: Path, chunk_size: tuple = (6000,8000)) -> list[str]:
        """
        
        """
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True
        pipeline_options.ocr_options.lang = ["es"]
        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=mp.cpu_count() - 1, device=AcceleratorDevice.AUTO
        )

        doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        chunks = []
        try:
            conv_result = doc_converter.convert(pdf_path)
            text = conv_result.document.export_to_markdown()
            # Extract overlap chunks from the markdown text based in the semmantic using semantic split
            splitter = MarkdownSplitter(chunk_size)
            chunks = splitter.chunks(text)
        except Exception as e:
            logtext.error(f"Error processing PDF file {pdf_path}: {e}")
        return chunks
    # def pdf2str(self, pdf_path: Path) -> str:
    #     try:
    #         with open(pdf_path, "rb") as f:
    #             pdf_content = f.read()
    #             pdf_file = io.BytesIO(pdf_content)
    #             reader = PyPDF2.PdfReader(pdf_file)
    #             text = ""
    #             for page in reader.pages:
    #                 text += page.extract_text() + "\n"
    #         return text
    #     except Exception as e:
    #         logtext.error(f"Error reading PDF file {pdf_path}: {e}")
    #         return ""
    
    def json2str(self, json_path: Path) -> tuple[str,str]:
        """
        
        """
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Try common field names for full text - can be improved in the future
            full_text_fields = ["full_text", "fulltext", "text", "content", "body"]
            title_fields = ["title", "Title", "article_title"]

            full_text = "FULL TEXT\n"
            title = ""

            for field in full_text_fields:
                if field in data and data[field]:
                    temp = data[field]
                    if isinstance(temp, dict):
                        for subfield in full_text_fields:
                            if subfield in temp and temp[subfield]:
                                full_text += str(temp[subfield])
                                break
                    else:
                        full_text += str(temp)
                    break

            for field in title_fields:
                if field in data and data[field]:
                    title = str(data[field])
                    break
            
            # Extract sections from full text
            if full_text:
                sections = self._extract_sections(full_text)
                # Combine all matched sections into one text with headings
                for k, v in sections.items():
                    if k == 'materials_text' and v != '':
                        full_text += '\nMATERIALS\n' + v
                    elif k == 'methods_text' and v != '':
                        full_text += '\nMETHODS\n' + v
                    elif k == 'data_text' and v != '':
                        full_text += '\nDATA AVAILABILITY\n' + v
                # full_text += '\n' + sections['combined_text'] + '\n'
            
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
                    "num_predict": 3000,  # max tokens predict for response
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

    def _run_llm_chunks(self, chunks: list[str], chunk_model: str = 'qwen3:14b') -> list[str]:
        response_parser = LLMResponseParser()
        chunks_selected = []
        for i, chunk in enumerate(chunks):
            logtext.info(f"Processing chunk {i+1}/{len(chunks)}")
            prompt = PromptTemplates()._chunk_genotype_prompt(chunk_text=chunk)
            response = ollama.generate(
                model = chunk_model,
                prompt=prompt,
                system=SYSTEM_PROMPT_GENOTYPE,
                # format="json",
                # raw=True,
                options={
                    "temperature": 0.1,  # for consistency
                    "top_k": 10, # more focused
                    "top_p": 0.5,  # more conservative
                    "num_predict": 500,  # max tokens predict for response
                    "stream": False,
                    "num_ctx": 8192,  # context window
                },
            )
            raw_text = response["response"] if isinstance(response, dict) else response.response
            logtext.info(f"Chunk {i+1} response: {raw_text}\n")
            answer = response_parser.parse_boolean_answer(raw_text)
            logtext.info(f"Parsed boolean answer: {answer}")
            if answer == True:
                chunks_selected.append(chunk)
                logtext.info(f"Chunk {i+1} selected for genotype information.\n")
            else:
                logtext.info(f"Chunk {i+1} NOT selected.\n")
        logtext.info(f"Total selected chunks: {len(chunks_selected)} out of {len(chunks)}")
        logtext.info(f"Selected chunks: {chunks_selected}")
        return chunks_selected

    def _run_llm_chunks_table(self, table: str, chunk_model: str = 'qwen3:14b') -> str:
        response_parser = LLMResponseParser()
        # selected = ''
        
        prompt = PromptTemplates()._chunk_genotype_tables_prompt(chunk_text=table)
        response = ollama.generate(
            model = chunk_model,
            prompt=prompt,
            system=SYSTEM_PROMPT_GENOTYPE,
            # format="json",
            # raw=True,
            options={
                "temperature": 0.1,  # for consistency
                "top_k": 10, # more focused
                "top_p": 0.5,  # more conservative
                "num_predict": 500,  # max tokens predict for response
                "stream": False,
                "num_ctx": 8192,  # context window
            },
        )
        raw_text = response["response"] if isinstance(response, dict) else response.response
        logtext.info(f"Response: {raw_text}\n")
        answer = response_parser.parse_boolean_answer(raw_text)
        logtext.info(f"Parsed boolean answer: {answer}")
        if answer == True:
            return table
        else:
            return ''
def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Extract biological data from academic articles')
    parser.add_argument('--input', type=Path, help='Directory containing PDF and JSON files')
    parser.add_argument('--output', type=Path, help='Directory to save results')
    parser.add_argument('--model', default='llama3.3:70b', help='Ollama model name. Available models can be listed with "ollama list" command. Actually we have llama3.1, llama3.3:70b and qwen3:14b')
    
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
        # if file_path.suffix.lower() == '.json':
            
        #     title, text = extractor.json2str(file_path)
        #     if text:
        #         logtext.info(f"Extracted text from {file_path.name} (Title: {title[:30]}...)")
        #         # logtext.info(f"Text snippet: {text}")
        #         logtext.info(f"Characters: {len(text)}")
        #         logtext.info(f"Words (approx): {len(text.split())}")
        #         logtext.info(f"Tokens (rough): {len(text.split()) * 1.3}")
        #         logtext.info(f"Running LLM extraction for Genotype data...")
                
        #         response = llm_extractor.run_llm(text=text, objective='genotype')
        #     else:
        #         logtext.warning(f"No text extracted from {file_path.name}")

        # if file_path.suffix.lower() == '.pdf':
        #     # extractor = ExtractText()
        #     text = extractor.pdf2str(file_path)
        #     if text:
        #         logtext.info(f"Extracted text from {file_path.name}")
        #         # logtext.info(f"Text snippet: {text}...")
        #         logtext.info(f"Characters: {len(text)}")
        #         logtext.info(f"Words (approx): {len(text.split())}")
        #         logtext.info(f"Tokens (rough): {len(text.split()) * 1.3}")
        #         logtext.info(f"Running LLM extraction for Genotype data...")
        #         # llm_extractor = ExtractInfo(model=args.model)
        #         response = llm_extractor.run_llm(text=text, objective='genotype')
        #     else:
        #         logtext.warning(f"No text extracted from {file_path.name}")
        if file_path.suffix.lower() == '.pdf':
            extractor = ExtractText()
            chunks = extractor.pdf2split(file_path)
            if chunks:
                logtext.info(f"Extracted text from {file_path.name}")
                # logtext.info(f"Text snippet: {text}...")
                logtext.info(f"Chunks: {len(chunks)}")
                logtext.info(f"Running LLM extraction for Genotype data...")
                # llm_extractor = ExtractInfo(model=args.model)
                response = llm_extractor._run_llm_chunks(chunks=chunks)
                if len(response) > 0:
                    logtext.info(f"Genotype-related chunks found: {len(response)}")
                    # join the selected chunks into one text
                    combined_text = "\n".join(response)
                    combined_text = LLMResponseParser().clean_text(combined_text)
                    logtext.info(f"Combined selected chunks length: {len(combined_text)} characters")
                    logtext.info(f"Appending tables extracted from PDF...\n{combined_text}")
                    tables = extractor._get_table_from_pdf(file_path)
                    if tables:
                        table = llm_extractor._run_llm_chunks_table(table=tables)
                        combined_text += "\n" + table
                        logtext.info(f"Table appended to combined text.")
                    logtext.info(f"New combined text length with tables: {len(combined_text)} characters")
                    logtext.info(f"Combined text snippet: {combined_text}\n")
                    logtext.info(f"Running final LLM extraction on combined genotype text...")
                    final_response = llm_extractor.run_llm(text=combined_text, objective='genotype')
                else:
                    logtext.info(f"No genotype-related chunks found in {file_path.name}")
            else:
                logtext.warning(f"No text extracted from {file_path.name}")


    # extractor = ExtractInfo(model=args.model)
if __name__ == "__main__":
    main()
