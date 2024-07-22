import PyPDF2

# Open the PDF file
with open('SQLNotes.pdf', 'rb') as file:
    reader = PyPDF2.PdfReader(file)
    
    text = ""
    for page in reader.pages:
        text += page.extract_text()

print(text)
