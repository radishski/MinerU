import os
import sys
import traceback
from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader, S3DataReader
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.config.enums import SupportedPdfParseMethod
import argparse

# Get the parent directory of the current script
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
# Add the parent directory to the system path
sys.path.append(parent_dir)



class ExtractPdfByMinerUProcessor:

    def __init__(self):
        pass

    def extract_pdf(self, pdf_file_path, output_dir):
        try:
            do_extract_pdf(pdf_file_path, output_dir)
            print(f"处理 {pdf_file_path} 完成")
            return f"ok"
        except Exception as e:
            print(f"解析 {pdf_file_path} 报错: {e}, \n {e.__traceback__}, \n {traceback.format_exc()}")
            return f"不ok"

def do_extract_pdf(pdf_file_path, output_dir):
    # name_without_suff = pdf_file_name.split(".")[0]
    pdf_file_name = os.path.basename(pdf_file_path)
    name_without_suff = pdf_file_name.replace('.pdf', '')
    local_image_dir = os.path.join(output_dir, name_without_suff, 'images')
    local_md_dir = os.path.join(output_dir, name_without_suff, 'md')

    # prepare env
    # image_dir = str(os.path.basename(local_image_dir))

    # os.makedirs(local_image_dir, exist_ok=True)

    image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(
        local_md_dir
    )

    image_writer = None

    # read bytes
    reader1 = FileBasedDataReader("")
    pdf_bytes = reader1.read(pdf_file_path)  # read the pdf content
    # proc
    ## Create Dataset Instance
    ds = PymuDocDataset(pdf_bytes, lang='en')

    ## inference
    if ds.classify() == SupportedPdfParseMethod.OCR:
        infer_result = ds.apply(doc_analyze, ocr=True, lang='en', contrast_alpha=1.2)

        ## pipeline
        pipe_result = infer_result.pipe_ocr_mode(image_writer)

    else:
        infer_result = ds.apply(doc_analyze, ocr=False)

        ## pipeline
        pipe_result = infer_result.pipe_txt_mode(image_writer)

    ### draw model result on each page
    infer_result.draw_model(os.path.join(local_md_dir, f"{name_without_suff}_model.pdf"))

    ### get model inference result
    model_inference_result = infer_result.get_infer_res()

    ### draw layout result on each page
    pipe_result.draw_layout(os.path.join(local_md_dir, f"{name_without_suff}_layout.pdf"))

    ### draw spans result on each page
    pipe_result.draw_span(os.path.join(local_md_dir, f"{name_without_suff}_spans.pdf"))

    ### get markdown content
    md_content = pipe_result.get_markdown(img_dir_or_bucket_prefix='')

    ### dump markdown
    pipe_result.dump_md(md_writer, f"{name_without_suff}.md", img_dir_or_bucket_prefix='')

    ### get content list content
    content_list_content = pipe_result.get_content_list(image_dir_or_bucket_prefix='')

    ### dump content list
    pipe_result.dump_content_list(md_writer, f"{name_without_suff}_content_list.json", image_dir_or_bucket_prefix='')

    ### get middle json
    middle_json_content = pipe_result.get_middle_json()

    ### dump middle json
    pipe_result.dump_middle_json(md_writer, f'{name_without_suff}_middle.json')

    ### 成功标记
    with open(get_pdf_success_file(pdf_file_path, output_dir), 'w'):
        pass

    return 'ok'

def get_pdf_files(input_dir):
    pdf_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".pdf"):
                absolute_path = os.path.abspath(os.path.join(root, file))
                pdf_files.append(absolute_path)
    return pdf_files

def get_pdf_success_file(pdf_file_path, output_dir):
    pdf_file_name_with_extension = os.path.basename(pdf_file_path)
    pdf_file_name, _ = os.path.splitext(pdf_file_name_with_extension)
    return os.path.join(output_dir, pdf_file_name, "success")

def split_list_into_batches(lst, batch_num):
    batch_size = len(lst) // batch_num
    remainder = len(lst) % batch_num
    batches = []
    start = 0
    for i in range(batch_num):
        end = start + batch_size
        if i < remainder:
            end += 1
        batches.append(lst[start:end])
        start = end
    return batches

if __name__ == '__main__':

    do_extract_pdf('/Users/zeyuan.zhang/Desktop/test_case.pdf', '/Users/zeyuan.zhang/Desktop/llm/pdf_extract_local')



