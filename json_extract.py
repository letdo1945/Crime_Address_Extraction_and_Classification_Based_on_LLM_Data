import csv
import re
import codecs


def extract_json_content(text):
    """从文本中提取```json和```之间的内容"""
    pattern = r'```json(.*?)```'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def process_csv(input_file, output_file):
    """处理CSV文件"""
    with open(input_file, 'r', encoding='utf-8-sig') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames

        with codecs.open(output_file, 'w', encoding='utf-8-sig') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()

            for row in reader:
                if 'LLM' in row and row['LLM']:
                    # 提取json内容并替换原output列
                    row['LLM'] = extract_json_content(row['LLM'])
                writer.writerow(row)


if __name__ == '__main__':
    input_filename = './output/LEC_demo3_processed_data.csv'  # 输入文件名
    output_filename = './output/LEC_demo3_processed_data_json.csv'  # 输出文件名

    process_csv(input_filename, output_filename)
    print(f"处理完成，结果已保存到 {output_filename}")