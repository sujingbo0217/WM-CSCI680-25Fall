import os
import re
import csv
from typing import List
from github import Github, GithubException
import javalang
from pydriller import Repository


GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")


ALLOWED_LICENSES = [
    'mit', 'apache-2.0', 'bsd-3-clause', 'bsd-2-clause', 'gpl', 'lgpl', 'mpl-2.0'
]

MIN_FILE_LINES = 3
MAX_FILE_LINES = 1_000

def check_repo_license(gh: Github, repo_url: str) -> bool:
    """Checks if a repository's license is in the allowed list."""
    repo_name = '/'.join(repo_url.split('/')[-2:])
    try:
        repo = gh.get_repo(repo_name)
        if repo.license and repo.license.key in ALLOWED_LICENSES:
            return True
        print(f"Skipping {repo_name}: Incompatible or missing license ('{repo.license.key if repo.license else 'None'}').")
        return False
    except GithubException as e:
        print(f"Could not access repo {repo_name}: {e}")
        return False
    

def clean_comments(code: str) -> str:
    """
    Removes single-line, multi-line, and Javadoc comments from Java code.
    """
    code = re.sub(r'//.*', '', code)
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    code = "\n".join(line for line in code.split('\n') if line.strip())
    return code


def gen_java_methods(repo_urls: List[str], csv_filepath: str, line_limit: int, split_type: str):
    headers = [
                'dataset_split', 
                'repo_name', 
                'repo_url',
                'commit_sha', 
                'file_path',
                'method_name', 
                'start_line', 
                'end_line', 
                'signature',
                'original_code', 
                'code_tokens',
    ]
    
    counter = 0
    all_methods = []

    for repo_url in repo_urls:
        repo_name = repo_url.split('/')[-1]
        print(f"Fetch Repo: {repo_name}")

        # if not GITHUB_TOKEN:
        #     print("Warning: GITHUB_TOKEN environment variable not set. API calls may be rate-limited.")
    
        # gh = Github(GITHUB_TOKEN)
        # if not check_repo_license(gh, repo_url):
        #     continue

        for commit in Repository(repo_url).traverse_commits():
            for file in commit.modified_files:
                is_template = 'archetype-resources' in file.new_path if file.new_path else False
                if file.filename.endswith('.java') and file.source_code and not is_template:
                    try:
                        tree = javalang.parse.parse(file.source_code)
                        for path, node in tree.filter(javalang.tree.MethodDeclaration):
                            code = file.source_code
                            lines = code.split('\n')
                            lines_without_imports = [line for line in lines if not line.strip().startswith('import ')]
                            code = '\n'.join(lines_without_imports)
                            # ------ data cleaning ------

                            # a. remove comments
                            code = clean_comments(code)
                            
                            # b. filter empty methods
                            if not code.strip():
                                continue
                            # c. filter too short or too long methods
                            num_lines = len(code.splitlines())
                            if not (MIN_FILE_LINES < num_lines < MAX_FILE_LINES):
                                continue
                            # d. filter files are not valid Java
                            try:
                                tokens = list(javalang.tokenizer.tokenize(code))
                                # e. filter files without actual code
                                if not tokens:
                                    continue
                                javalang.parse.parse(code)
                            except (javalang.parser.JavaSyntaxError, javalang.tokenizer.LexerError):
                                continue

                            # 1. get start and end line number
                            start = node.position.line if node.position else None
                            # print(f"Start line: {start}")
                            
                            # if node.body:
                            #     last_stmt = node.body.statements[-1]
                            #     if last_stmt.position and last_stmt.position.line:
                            #         end = max(end, last_stmt.position.line)

                            if not start:
                                continue

                            end = start + len(code.splitlines()) - 1

                            # 2. tokenize method source code
                            tokens = javalang.tokenizer.tokenize(code)
                            code_tokens = ', '.join([tok.value for tok in tokens])

                            # 3. reconstruct the method signature
                            params = ', '.join([f"{param.type.name} {param.name}" for param in node.parameters])
                            signature = f"{node.return_type.name if node.return_type else 'void'} {node.name}({params})"

                            # 4. input data to csv file
                            data = {
                                    "dataset_split": split_type,
                                    "repo_name": repo_name,
                                    "repo_url": repo_url,
                                    "commit_sha": commit.hash,
                                    "file_path": file.new_path,
                                    "method_name": node.name,
                                    "start_line": start,
                                    "end_line": end,
                                    "signature": signature,
                                    "original_code": code,
                                    "code_tokens": code_tokens,
                            }
                            
                            # print(data)
                            all_methods.append(data)
                            counter += 1
                            
                            if counter >= line_limit // 4 or len(all_methods) >= line_limit:
                                break

                            """
                            print(f"\nCommit: {commit.hash}")
                            print(f"File: {file.filename}")
                            print(f"Method: {node.name}")
                            print(f"Tokens: {code_tokens[:20]} ...")  # preview
                            """

                        # print(f"code: {code[:30]}...")  # preview
                        
                        if counter >= line_limit // 4 or len(all_methods) >= line_limit:
                            break
                    except:
                        # print(f"Could not parse {file.new_path} in commit {commit.hash}")
                        continue
                if counter >= line_limit // 4 or len(all_methods) >= line_limit:
                    break
            if counter >= line_limit // 4:
                counter = 0
                break
            if len(all_methods) >= line_limit:
                break
        
    print(f"Found {len(all_methods)} methods. Writing to CSV...")
    
    with open(csv_filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f,
                                fieldnames=headers,
                                quoting=csv.QUOTE_NONE,
                                escapechar='\\')
        writer.writeheader()
        writer.writerows(all_methods)
    
    print(f"Successfully created {csv_filepath}!")


if __name__ == "__main__":
    train_repo_urls = [
        "https://github.com/apache/kafka",
        "https://github.com/apache/maven",
        "https://github.com/quarkusio/quarkus",
        "https://github.com/spring-projects/spring-boot",
    ]
    eval_repo_urls = [
        "https://github.com/gradle/gradle",
        "https://github.com/netty/netty",
        "https://github.com/micronaut-projects/micronaut-core",
        "https://github.com/YunaiV/ruoyi-vue-pro",
    ]
    
    train_output_file = "data/java_train_data.csv"
    eval_output_file = "data/java_eval_data.csv"

    gen_java_methods(train_repo_urls, train_output_file, line_limit=20_000, split_type="train")
    gen_java_methods(eval_repo_urls, eval_output_file, line_limit=5_000, split_type="eval")
