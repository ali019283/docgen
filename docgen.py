import os
import tiktoken
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

prompteng = """You are an expert code documenter. Add detailed documentation comments above each function, class, and method in ALL the following files.

    RULES:
    1. Add VERY DETAILED comments DIRECTLY above each method or action. Add it to every function no matter if its useful or not
    2. For EACH function/method include:
    - What it does
    - All parameters (type and description, explain params briefly with examples too)
    - Return value (type and description)
    - Comprehensive example usage
    3. For EACH class/variables include:
    - Its purpose
    - Key methods overview
    - All elements purpose
    4. Use the EXACT comment style for each language
    5. PRESERVE all existing code exactly
    6. RETURN the COMPLETE files with your additions
    7. It must be a COMPLETE documentation of func/class/var or whatever you find helpfull to document. It must be very detailed and i must be able to understand it fully and use it in a development enviroment just from reading the comments you made.
    8. Give ALL the files in response with some explanation, no leaving ANY files behind. Return the WHOLE file and a valid explanation for ALL the files i give.
    9. In the end, make a file named DOCGEN_DOCUMENT.md and document the whole project using markdown. DONT SKIP.
    10. Format response like this given below, when giving path, give the exact path dont give it all non-capital or dont convert numbers into letters, RAW TEXT ONLY:

    === FILENAME: path/to/file.ext ===
    // Existing license header...
    /* NEW VERY DETAILED DOC COMMENT */
    function originalCode() { ... }

    === FILENAME: next/file.ext ===
    ..."""

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
PROJECT_DIR = Path(os.getenv("PROJECT_DIR", ".")).resolve()
MODEL = "deepseek/deepseek-r1"
MAX_TOKENS = 128000
MAX_RESPONSE_TOKENS = 64000
EXTS = [".py", ".js", ".ts", ".cpp", ".c", ".h", ".java", ".rb", ".go", ".rs", ".cs", ".md"]
MAX_FILES_PER_BATCH = 5

tokenizer = tiktoken.get_encoding("cl100k_base")

oai = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    default_headers={
        "X-Title": "docgen"
    }
)

def is_relative_to(path, base):
    try:
        path.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False

def gather_code_files(root_dir):
    files = []
    root_dir = Path(root_dir).resolve()
    for path in root_dir.rglob("*"):
        if path.suffix.lower() in EXTS and path.is_file():
            try:
                text = path.read_text(encoding="utf-8")
                if text.strip():
                    files.append({
                        'path': path,
                        'content': text
                    })
            except Exception as e:
                print(f"skip {path}: {e}")
    print(f"Found {len(files)} files to process:")
    for f in files:
        print(f" - {f['path']}")
    return files

def create_batches(files, max_tokens):
    batches = []
    current_batch = []
    current_tokens = 0
    
    for file in files:
        file_tokens = len(tokenizer.encode(file['content']))
        if file_tokens > max_tokens * 0.8:
            print(f"Warning: File too large - {file['path']} ({file_tokens} tokens)")
            continue
            
        if (current_tokens + file_tokens > max_tokens * 0.8 or 
            len(current_batch) >= MAX_FILES_PER_BATCH):
            batches.append(current_batch)
            current_batch = [file]
            current_tokens = file_tokens
        else:
            current_batch.append(file)
            current_tokens += file_tokens
    
    if current_batch:
        batches.append(current_batch)
    
    print(f"\nCreated {len(batches)} batches with max {MAX_FILES_PER_BATCH} files per batch")
    for i, batch in enumerate(batches):
        batch_tokens = sum(len(tokenizer.encode(f['content'])) for f in batch)
        print(f"Batch {i}: {len(batch)} files, ~{batch_tokens} tokens")
    
    return batches

def generate_prompt(files, prompt_template):
    prompt = prompt_template + "\n\nPlease document the following files:\n"
    for file in files:
        try:
            rel_path = file['path'].resolve().relative_to(PROJECT_DIR)
            prompt += f"\n=== FILENAME: {rel_path} ===\n"
        except ValueError:
            prompt += f"\n=== FILENAME: {file['path']} ===\n"
        prompt += file['content']
    
    prompt += "\n\nDOCUMENT ALL FILES FOLLOWING THE RULES EXACTLY."
    return prompt

def parse_response(response, original_files):
    def normalize_path(p):
        try:
            path = Path(p)
            try:
                rel_path = path.resolve().relative_to(PROJECT_DIR.resolve())
                return str(rel_path).lower().replace('\\', '/')
            except ValueError:
                return str(path).lower().replace('\\', '/')
        except:
            return str(p).lower().replace('\\', '/')

    response_files = {}
    current_file = None
    current_content = []
    
    for line in response.split('\n'):
        if line.startswith("=== FILENAME:"):
            if current_file and current_content:
                response_files[current_file] = '\n'.join(current_content).strip()
            path_str = line.split('=== FILENAME:')[1].split('===')[0].strip()
            current_file = normalize_path(path_str)
            current_content = []
        elif current_file is not None:
            current_content.append(line)
    
    if current_file and current_content:
        response_files[current_file] = '\n'.join(current_content).strip()

    print(f"\nFound {len(response_files)} files in AI response:")
    for path in response_files:
        print(f" - {path}")

    results = []
    for orig_file in original_files:
        orig_path = orig_file['path']
        orig_normalized = normalize_path(orig_path)
        found = False
        for resp_path, resp_content in response_files.items():
            if resp_path == orig_normalized:
                results.append({
                    'path': orig_path,
                    'content': resp_content
                })
                found = True
                break
        
        if not found:
            orig_parent = str(Path(orig_normalized).parent).lower()
            orig_filename = Path(orig_normalized).name.lower()
            for resp_path, resp_content in response_files.items():
                resp_filename = Path(resp_path).name.lower()
                if resp_filename == orig_filename:
                    resp_parent = str(Path(resp_path).parent).lower()
                    if orig_parent in resp_parent or resp_parent in orig_parent:
                        print(f"Matched by similar path: {orig_path}")
                        results.append({
                            'path': orig_path,
                            'content': resp_content
                        })
                        found = True
                        break
        
        if not found:
            orig_filename = Path(orig_normalized).name.lower()
            for resp_path, resp_content in response_files.items():
                if Path(resp_path).name.lower() == orig_filename:
                    print(f"Matched by filename only: {orig_filename}")
                    results.append({
                        'path': orig_path,
                        'content': resp_content
                    })
                    found = True
                    break
        
        if not found:
            print(f"Warning: File not matched - {orig_path}")
            results.append({
                'path': orig_path,
                'content': orig_file['content']
            })
    
    return results

def call_ai(prompt):
    try:
        print(f"Sending prompt with {len(tokenizer.encode(prompt))} tokens...")
        response = oai.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=MAX_RESPONSE_TOKENS
        )
        content = response.choices[0].message.content
        print(f"Received response with {len(tokenizer.encode(content))} tokens")
        return content
    except Exception as e:
        print(f"API Error: {e}")
        return None

def main():
    doc_path = PROJECT_DIR / "DOCGEN_DOCUMENTATION.md"
    with open(doc_path, 'w', encoding='utf-8') as f:
        f.write("# Project Documentation\n\n")
    
    print(f"Documenting files in: {PROJECT_DIR}")
    files = gather_code_files(PROJECT_DIR)
    
    if not files:
        print("No !!!")
        return
    
    batches = create_batches(files, MAX_TOKENS)
    print(f"\nProcessing {len(batches)} batches...")
    
    all_processed_files = []
    for i, batch in enumerate(batches):
        print(f"\nProcessing batch {i+1}/{len(batches)} with {len(batch)} files...")
        prompt = generate_prompt(batch, prompteng)
        
        with open(f"batch_{i}_prompt.txt", 'w') as f:
            f.write(prompt)
        
        response = call_ai(prompt)
        if response:
            with open(f"batch_{i}_response.txt", 'w') as f:
                f.write(response)
            
            processed_files = parse_response(response, batch)
            all_processed_files.extend(processed_files)
        else:
            print("Skipping batch")
            all_processed_files.extend(batch)
    
    print("\nSaving files...")
    for file in all_processed_files:
        try:
            with open(file['path'], 'w', encoding='utf-8') as f:
                f.write(file['content'])
            print(f"Updated {file['path']}")
        except Exception as e:
            print(f"Failed to write {file['path']}: {e}")
    
    print("\nDocumentation process completed!")

if __name__ == "__main__":
    main()