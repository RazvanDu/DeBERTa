with open('deberta_embd_saved.tsv', 'r', encoding='utf-8') as file:
    file_content = file.read()
    print('Read from file:', file_content)
    print('Hexadecimal:', file_content.encode('utf-8').hex())