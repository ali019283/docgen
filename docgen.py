import os
import tiktoken
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from alive_progress import alive_bar

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
PROJECT_DIR = os.getenv("PROJECT_DIR", ".")
#MODEL = "deepseek/deepseek-r1:free"
MODEL = "deepseek/deepseek-chat:free" # a lot faster but use the one above if u want nun mistakes
INPUT_TOKEN_LIMIT = 5000
MAX_RESPONSE_TOKENS = 5000

EXTS = [".py", ".js", ".ts", ".cpp", ".c", ".h", ".java", ".rb", ".go", ".rs", ".cs", ".xml", ".md", ".swift", ".vb", ".php", ".css", ".html", ".txt", ".json", ".yaml", ".sh", ".yml"]

oai = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    default_headers={
        "X-Title": "docgen"
    }
)

def gather_code_content(root_dir):
    content = ""
    for path in Path(root_dir).rglob("*"):
        if path.suffix.lower() in EXTS and path.is_file():
            try:
                text = path.read_text(encoding="utf-8")
                if text.strip():
                    content += f"\n\n=== {path.relative_to(root_dir)} ===\n{text}"
            except Exception as e:
                print(f"skip {path}: {e}")
    return content

def split_into_chunks(text, max_tokens):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk = tokens[i:i+max_tokens]
        chunks.append(tokenizer.decode(chunk))
    return chunks

def call_ai(messages):
    try:
        response = oai.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=MAX_RESPONSE_TOKENS
        )
        return response.choices[0].message.content
    except Exception as e:
        print(e)
        return None

def main():
    print(f"working on dir: {PROJECT_DIR}")
    full_code = gather_code_content(PROJECT_DIR)
    #print({len(full_code)})
    tokenizer = tiktoken.get_encoding("cl100k_base")
    total_tokens = len(tokenizer.encode(full_code))
    #print(total_tokens)
    chunks = split_into_chunks(full_code, INPUT_TOKEN_LIMIT)
    all_outputs = [""]
    with alive_bar(len(chunks)) as bar:
        for i, chunk in enumerate(chunks):
            #print(f"\nchunk {i+1}/{len(chunks)}")        
            bar()
            user_prompt = f"""This is the analysis by the last prompt for chunk {i}:
            {all_outputs[i]}

            ====

            You are an expert technical writer. Help generate developer documentation. Analyze the chunk i am about to send you and merge it with the last prompt you generated (Given above). Dont leave anything out.

            Analyze this segment of a large codebase and generate detailed developer documentation (Markdown format). Include:
            - Key functions/classes with parameters
            - Detailed usage examples
            - Architecture insights

            Treat this as a documentation for a very serious project. So it should be very organized, and very explaining.

            Chunk {i+1}:
            {chunk}
            """
            messages =[{"role": "user", "content": user_prompt}]
            response = call_ai(messages)
            all_outputs.append(response)

    with open(f"FINAL_DOC.md", "w", encoding="utf-8") as f:
        f.write(all_outputs[-1])
    print(f"FINAL_DOC.md")

if __name__ == "__main__":
    main()
