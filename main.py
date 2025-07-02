import os
import logging
import requests
import fitz
import asyncio
import json
import httpx
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
import ollama

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI()


class URLRequest(BaseModel):
    url: str


@app.get("/health")
def health_check():
    return {"status": "ok", "message": "FastAPI backend is running!"}


@app.post("/summarize_arxiv/")
async def summarize_arxiv(request: URLRequest):
    """Downloads an Arxiv PDF, extracts text, and summarizes it using Ollama (Gemma 3) in parallel."""
    try:
        url = request.url
        logger.info("---------------------------------------------------------")
        logger.info(f"Downloading PDF from: {url}")

        pdf_path = download_pdf(url)
        if not pdf_path:
            return {"error": "Failed to download PDF. Check the URL."}

        logger.info(f"PDF saved at: {pdf_path}")

        # Extract text from the PDF
        text = extract_text_from_pdf(pdf_path)
        if not text:
            return {"error": "No text extracted from PDF"}

        logger.info(f"Extracted text length: {len(text)} characters")
        logger.info("---------------------------------------------------------")

        # Summarize extracted text in parallel
        summary = await summarize_text_parallel(text)
        logger.info("Summarization complete")

        return {"summary": summary}

    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        return {"error": "Failed to process PDF"}


def download_pdf(url):
    """Downloads a PDF from a given URL and saves it locally."""
    try:
        if not url.startswith("https://arxiv.org/pdf/"):
            logger.error(f"Invalid URL: {url}")
            return None  # Prevents downloading non-Arxiv PDFs

        response = requests.get(url, timeout=30)  # Set timeout to prevent long waits
        if response.status_code == 200 and "application/pdf" in response.headers.get("Content-Type", ""):
            pdf_filename = "arxiv_paper.pdf"
            with open(pdf_filename, "wb") as f:
                f.write(response.content)
            return pdf_filename
        else:
            logger.error(f"Failed to download PDF: {response.status_code} (Not a valid PDF)")
            return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading PDF: {e}")
        return None


def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file using PyMuPDF (faster than Unstructured PDFLoader)."""
    try:
        doc = fitz.open(pdf_path)
        text = "\n".join([page.get_text("text") for page in doc])
        return text
    except Exception as e:
        logger.error(f"Error extracting text: {e}")
        return ""

async def summarize_chunk_with_retry(chunk, chunk_id, total_chunks, max_retries=2):
    """Retry mechanism wrapper for summarize_chunk_wrapper."""
    retries = 0
    while retries  0:
                logger.info(f" Retry attempt {retries}/{max_retries} for chunk {chunk_id}/{total_chunks}")

            result = await summarize_chunk_wrapper(chunk, chunk_id, total_chunks)

            # If the result starts with "Error", it means there was an error but no exception was thrown
            if isinstance(result, str) and result.startswith("Error"):
                logger.warning(f" Soft error on attempt {retries+1}/{max_retries+1} for chunk {chunk_id}: {result}")
                retries += 1
                if retries  0:
                    logger.info(f" Successfully processed chunk {chunk_id} after {retries} retries")
                return result

        except Exception as e:
            retries += 1
            logger.error(f" Exception on attempt {retries}/{max_retries+1} for chunk {chunk_id}: {str(e)}")

            if retries &lt;= max_retries:
                # Exponential backoff
                wait_time = 5 * (2 ** (retries - 1))
                logger.info(f&quot; Waiting {wait_time}s before retry for chunk {chunk_id}&quot;)
                await asyncio.sleep(wait_time)
            else:
                logger.error(f&quot; All retry attempts exhausted for chunk {chunk_id}&quot;)
                return f&quot;Error processing chunk {chunk_id} after {max_retries+1} attempts: {str(e)}&quot;

    # This should never be reached, but just in case
    return f&quot;Error: Unexpected end of retry loop for chunk {chunk_id}&quot;


async def summarize_text_parallel(text):
    &quot;&quot;&quot;Process text in chunks optimized for Gemma 3&#039;s 128K context window with full parallelism and retry logic.&quot;&quot;&quot;
    token_estimate = len(text) // 4
    logger.info(f&quot; Token Estimate: {token_estimate}&quot;)

    # Use larger chunks since Gemma 3 can handle 128K tokens
    chunk_size = 10000 * 4  # Approximately 32K tokens per chunk
    chunk_overlap = 100   # Larger overlap to maintain context

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[&quot;\n\n&quot;, &quot;\n&quot;, &quot;. &quot;, &quot; &quot;, &quot;&quot;]
    )
    chunks = splitter.split_text(text)
    logger.info(&quot;---------------------------------------------------------&quot;)
    logger.info(f&quot; Split text into {len(chunks)} chunks&quot;)
    logger.info(&quot;---------------------------------------------------------&quot;)

    # Log chunk details
    for i, chunk in enumerate(chunks, 1):
        chunk_length = len(chunk)
        logger.info(f&quot; Length: {chunk_length} characters ({chunk_length // 4} estimated tokens)&quot;)

    logger.info(&quot;---------------------------------------------------------&quot;)
    logger.info(f&quot; Processing {len(chunks)} chunks in parallel with retry mechanism...&quot;)

    # Create tasks for each chunk with retry mechanism
    tasks = [summarize_chunk_with_retry(chunk, i+1, len(chunks), max_retries=2) for i, chunk in enumerate(chunks)]

    # Process chunks with proper error handling at the gather level
    try:
        # Using return_exceptions=True to prevent one failure from canceling all tasks
        summaries = await asyncio.gather(*tasks, return_exceptions=True)

        # Process the results, handling any exceptions
        processed_summaries = []
        for i, result in enumerate(summaries):
            if isinstance(result, Exception):
                # An exception was returned
                logger.error(f&quot; Task for chunk {i+1} returned an exception: {str(result)}&quot;)
                processed_summaries.append(f&quot;Error processing chunk {i+1}: {str(result)}&quot;)
            else:
                # Normal result
                processed_summaries.append(result)

        summaries = processed_summaries

    except Exception as e:
        logger.error(f&quot; Critical error in gather operation: {str(e)}&quot;)
        return f&quot;Critical error during processing: {str(e)}&quot;

    logger.info(&quot; All chunks processed (with or without errors)&quot;)

    # Check if we have at least some successful results
    successful_summaries = [s for s in summaries if not (isinstance(s, str) and s.startswith(&quot;Error&quot;))]
    if not successful_summaries:
        logger.warning(&quot; No successful summaries were generated.&quot;)
        return &quot;No meaningful summary could be generated. All chunks failed processing.&quot;

    # Combine summaries with section markers, including error messages for failed chunks
    combined_chunk_summaries = &quot;\n\n&quot;.join(f&quot;Section {i+1}:\n{summary}&quot; for i, summary in enumerate(summaries))
    logger.info(f&quot; Combined summaries length: {len(combined_chunk_summaries)} characters&quot;)
    logger.info(&quot; Generating final summary...&quot;)

    # Create final summary with system message
    final_messages = [
        {
            &quot;role&quot;: &quot;system&quot;,
            &quot;content&quot;: &quot;You are a technical documentation writer. Focus ONLY on technical details, implementations, and results. DO NOT mention papers, citations, or authors.&quot;
        },
        {
            &quot;role&quot;: &quot;user&quot;,
            &quot;content&quot;: f&quot;&quot;&quot;Create a comprehensive technical document focusing ONLY on the implementation and results.
            Structure the content into these sections:

            1. System Architecture
            2. Technical Implementation
            3. Infrastructure &amp; Setup
            4. Performance Analysis
            5. Optimization Techniques

            CRITICAL INSTRUCTIONS:
            - Focus ONLY on technical details and implementations
            - Include specific numbers, metrics, and measurements
            - Explain HOW things work
            - DO NOT include any citations or references
            - DO NOT mention other research or related work
            - Some sections may contain error messages - please ignore these and work with available information

            Content to organize:
            {combined_chunk_summaries}
            &quot;&quot;&quot;
        }
    ]

    # Use async http client for the final summary with retry logic
    max_retries = 2
    retry_count = 0
    final_response = None

    while retry_count &lt;= max_retries:
        try:
            # Use async http client for the final summary as well
            payload = {
                &quot;model&quot;: &quot;gemma3:27b&quot;,
                &quot;messages&quot;: final_messages,
                &quot;stream&quot;: False
            }

            logger.info(f&quot; Sending final summary request (attempt {retry_count+1}/{max_retries+1})&quot;)
            # Make async HTTP request with increased timeout for final summary
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    &quot;http://localhost:11434/api/chat&quot;,
                    json=payload,
                    timeout=httpx.Timeout(connect=60, read=3600, write=60, pool=60)  # 15-minute read timeout
                )

                logger.info(f&quot; Received final summary response, status code: {response.status_code}&quot;)

                if response.status_code != 200:
                    raise Exception(f&quot;API returned non-200 status code: {response.status_code} - {response.text}&quot;)

                final_response = response.json()
                break  # Success, exit retry loop

        except Exception as e:
            retry_count += 1
            logger.error(f&quot; Error generating final summary (attempt {retry_count}/{max_retries+1}): {str(e)}&quot;)

            if retry_count &lt;= max_retries:
                # Exponential backoff
                wait_time = 10 * (2 ** (retry_count - 1))
                logger.info(f&quot; Waiting {wait_time}s before retrying final summary generation&quot;)
                await asyncio.sleep(wait_time)
            else:
                logger.error(f&quot; All retry attempts for final summary failed&quot;)
                return &quot;Failed to generate final summary after multiple attempts. Please check the logs for details.&quot;

    if not final_response:
        return &quot;Failed to generate final summary. Please check the logs for details.&quot;

    logger.info(&quot; Final summary generated&quot;)
    logger.info(f&quot; Final summary length: {len(final_response[&#039;message&#039;][&#039;content&#039;])} characters&quot;)
    return final_response[&#039;message&#039;][&#039;content&#039;]

async def summarize_chunk_wrapper(chunk, chunk_id, total_chunks):
    &quot;&quot;&quot;Asynchronous wrapper for summarizing a single chunk using Ollama via async httpx.&quot;&quot;&quot;
    logger.info(&quot;---------------------------------------------------------&quot;)
    logger.info(f&quot; Starting processing of chunk {chunk_id}/{total_chunks}&quot;)
    try:
        # Add system message to better control output
        messages = [
            {&quot;role&quot;: &quot;system&quot;, &quot;content&quot;: &quot;Extract only technical details. No citations or references.&quot;},
            {&quot;role&quot;: &quot;user&quot;, &quot;content&quot;: f&quot;Extract technical content: {chunk}&quot;}
        ]

        # Use httpx for truly parallel API calls
        payload = {
            &quot;model&quot;: &quot;gemma3:27b&quot;,
            &quot;messages&quot;: messages,
            &quot;stream&quot;: False
        }

        # Add better timeout and error handling
        try:
            # Make async HTTP request directly to Ollama API
            async with httpx.AsyncClient(timeout=3600) as client:  # Increased timeout to 10 minutes
                logger.info(f&quot; Sending request for chunk {chunk_id}/{total_chunks} to Ollama API - Gemma3 &quot;)
                response = await client.post(
                    &quot;http://localhost:11434/api/chat&quot;,  # Default Ollama API endpoint
                    json=payload,
                    # Adding connection timeout and timeout parameters
                    timeout=httpx.Timeout(connect=60, read=3600, write=60, pool=60)
                )
                logger.info(&quot;---------------------------------------------------------&quot;)
                logger.info(f&quot; Received response for chunk {chunk_id}/{total_chunks}, status code: {response.status_code}&quot;)

                if response.status_code != 200:
                    error_msg = f&quot;Ollama API error: {response.status_code} - {response.text}&quot;
                    logger.error(error_msg)
                    return f&quot;Error processing chunk {chunk_id}: API returned status code {response.status_code}&quot;

                response_data = response.json()
                summary = response_data[&#039;message&#039;][&#039;content&#039;]

            logger.info(f&quot; Completed chunk {chunk_id}/{total_chunks}&quot;)
            logger.info(f&quot; Summary length: {len(summary)} characters&quot;)
            logger.info(&quot;---------------------------------------------------------&quot;)
            return summary

        except httpx.TimeoutException as te:
            error_msg = f&quot;Timeout error for chunk {chunk_id}: {str(te)}&quot;
            logger.error(error_msg)
            return f&quot;Error in chunk {chunk_id}: Request timed out after 30 minutes. Consider increasing the timeout or reducing chunk size.&quot;

        except httpx.ConnectionError as ce:
            error_msg = f&quot;Connection error for chunk {chunk_id}: {str(ce)}&quot;
            logger.error(error_msg)
            return f&quot;Error in chunk {chunk_id}: Could not connect to Ollama API. Check if Ollama is running correctly.&quot;

    except Exception as e:
        # Capture and log the full exception details
        import traceback
        error_details = traceback.format_exc()
        logger.error(f&quot; Error processing chunk {chunk_id}: {str(e)}&quot;)
        logger.error(f&quot;Traceback: {error_details}&quot;)
        return f&quot;Error processing chunk {chunk_id}: {str(e)}&quot;


# Keep this function as a reference or remove it as it&#039;s been replaced by summarize_chunk_wrapper
def summarize_chunk(chunk, chunk_id):
    &quot;&quot;&quot;Summarizes a single chunk using Ollama (Gemma 3 LLM).&quot;&quot;&quot;
    logger.info(f&quot;\n{&#039;=&#039; * 40} Processing Chunk {chunk_id} {&#039;=&#039; * 40}&quot;)
    logger.info(f&quot; Input chunk length: {len(chunk)} characters&quot;)

    prompt = f&quot;&quot;&quot;
    You are a technical content extractor. Extract and explain ONLY the technical details from this section.

    Focus on:
    1. **System Architecture** – Design, component interactions, algorithms, configurations.
    2. **Implementation** – Code/pseudocode, data structures, formulas (with explanations), parameter values.
    3. **Experiments** – Hardware (GPUs, RAM), software versions, dataset size, training hyperparameters.
    4. **Results** – Performance metrics (accuracy, latency, memory usage), comparisons.

    **Rules:**
    - NO citations, references, or related work.
    - NO mention of authors or external papers.
    - ONLY technical details, numbers, and implementations.

    Text to analyze:
    {chunk}
    &quot;&quot;&quot;
    try:
        logger.info(f&quot; Sending chunk {chunk_id} to Ollama...&quot;)
        response = ollama.chat(model=&quot;gemma3:27b&quot;, messages=[{&quot;role&quot;: &quot;user&quot;, &quot;content&quot;: prompt}])
        summary = response[&#039;message&#039;][&#039;content&#039;]
        logger.info(f&quot; Successfully processed chunk {chunk_id}&quot;)
        logger.info(f&quot; Summary length: {len(summary)} characters&quot;)
        print(summary)
        return summary
    except Exception as e:
        logger.error(f&quot; Error summarizing chunk {chunk_id}: {e}&quot;)
        return f&quot;Error summarizing chunk {chunk_id}&quot;


if __name__ == &quot;__main__&quot;:
    import uvicorn

    logger.info(&quot;Starting FastAPI server on http://localhost:8000&quot;)
    uvicorn.run(app, host=&quot;0.0.0.0&quot;, port=8000, log_level=&quot;info&quot;)