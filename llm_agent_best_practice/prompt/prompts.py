from importlib import resources


class Prompts:
    def __init__(self):
        self.dict: dict[str, str] = {}
        with resources.files(__package__) as package_files:
            for file in package_files.iterdir():
                if file.suffix == '.txt':
                    with resources.as_file(file) as txt_file_path:
                        with open(txt_file_path, 'r', encoding='utf-8') as f:
                            self.dict[txt_file_path.name] = f.read()

    def get(self, key: str) -> str:
        return self.dict[key + '.txt']
