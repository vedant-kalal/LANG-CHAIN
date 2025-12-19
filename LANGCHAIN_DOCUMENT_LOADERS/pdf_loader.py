from langchain_community.document_loaders import PyPDFLoader

# | Use Case                         | Recommended Loader                              |
# |----------------------------------|-------------------------------------------------|
# | Simple, clean PDFs               | PyPDFLoader                                     |
# | PDFs with tables/columns         | PDFPlumberLoader                                |
# | Scanned/image PDFs               | UnstructuredPDFLoader or AmazonTextractPDFLoader|
# | Need layout and image data       | PyMuPDFLoader                                   |
# | Want best structure extraction   | UnstructuredPDFLoader                           |

# detailed code for all the above mentioned loaders :- https://docs.langchain.com/oss/python/integrations/document_loaders#pdfs

pdf = "LANGCHAIN_DOCUMENT_LOADERS/data_files/TASK - Hospital Case Management System (SQLAlchemy ORM).pdf"
loader = PyPDFLoader(pdf)
data = loader.load()
print(data)
print(len(data))
print(data[0].page_content)
print(data[0].metadata)