# Lab-00 Mining Java Methods

> Author: Bob Su

## 1. Environment Setup

```shell
pip install pandas javalang pydriller
```

## 2. Run Java methods generator

```shell
python data_generator.py
```

## 3. Code description

0. Check repo license using `PyGithub`

```python
gh = Github(GITHUB_TOKEN)
if not check_repo_license(gh, repo_url):
    continue
```

1. Get Java code from `git commit` of GitHub Repo using `pydriller`

```python
for commit in Repository(repo_url).traverse_commits():
    for file in commit.modified_files:
        is_template = 'archetype-resources' in file.new_path if file.new_path else False
        if file.filename.endswith('.java') and file.source_code and not is_template:
```

2. Parse Java code using `javalang`

```python
# 1. get start and end line number
start = node.position.line if node.position else None
end = start
if node.body:
    last_stmt = node.body.statements[-1]
    if last_stmt.position and last_stmt.position.line:
        end = max(end, last_stmt.position.line)
if not start or not end:
    continue
# 2. tokenize method source code
tokens = javalang.tokenizer.tokenize(code)
code_tokens = ', '.join([tok.value for tok in tokens])
# 3. reconstruct the method signature
params = ', '.join([f"{param.type.name} {param.name}" for param in node.parameters])
signature = f"{node.return_type.name if node.return_type else 'void'} {node.name}({params})"
# 4. input data to csv file
data = {
        "dataset_split": "train",
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
```

4. Data cleaning

Get rid of methods are too short/long, without code, or not valid Java code.

```python
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
```
