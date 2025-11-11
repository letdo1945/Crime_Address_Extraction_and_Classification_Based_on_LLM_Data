import pandas as pd
import json
import re
from difflib import SequenceMatcher
from typing import Dict, List, Tuple, Any
import os
from collections import defaultdict


def similarity(a: str, b: str) -> float:
    """计算两个字符串的相似度"""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, str(a), str(b)).ratio()


def safe_string_convert(value: Any) -> str:
    """安全地将值转换为字符串"""
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return str(value)
    return str(value)


def parse_crime_data(json_str: Any) -> List[Tuple[str, str]]:
    """解析犯罪数据，返回(地址, 分类)列表"""
    try:
        if pd.isna(json_str) or json_str is None or json_str == "":
            return []

        # 处理可能的JSON字符串格式问题
        if isinstance(json_str, str):
            # 尝试解析JSON字符串
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                # 如果不是有效的JSON，尝试其他解析方式
                print(f"JSON解析失败，原始数据: {json_str[:100]}...")
                return []
        else:
            data = json_str

        results = []

        # 处理不同的数据结构
        if isinstance(data, dict):
            for crime_key, crime_data in data.items():
                if isinstance(crime_data, dict):
                    address = safe_string_convert(crime_data.get('address', ''))
                    category = safe_string_convert(crime_data.get('address_category', ''))

                    # 确保category是字符串而不是列表
                    if isinstance(category, list):
                        category = category[0] if category else ""

                    if address and address not in ["", "null", "None"]:
                        results.append((address, category))
                else:
                    print(f"警告: crime_data不是字典类型: {type(crime_data)}")
        elif isinstance(data, list):
            # 处理列表形式的数据
            for item in data:
                if isinstance(item, dict):
                    address = safe_string_convert(item.get('address', ''))
                    category = safe_string_convert(item.get('address_category', ''))

                    if isinstance(category, list):
                        category = category[0] if category else ""

                    if address and address not in ["", "null", "None"]:
                        results.append((address, category))
        else:
            print(f"警告: 未知的数据结构类型: {type(data)}")

        return results

    except Exception as e:
        print(f"解析犯罪数据错误: {e}")
        print(f"问题数据: {str(json_str)[:200]}...")
        return []


def calculate_category_metrics(metrics: Dict) -> Dict:
    """计算每个分类的详细指标（Precision, Recall, F1）"""

    category_metrics = {}

    for category, stats in metrics['category_details'].items():
        if stats['total_val'] > 0:
            # 提取指标
            tp_extraction = stats['address_correct']  # 正确提取的地址数
            fp_extraction = max(0, stats['total_test'] - tp_extraction)  # 错误提取的地址数
            fn_extraction = max(0, stats['total_val'] - tp_extraction)  # 漏提取的地址数

            precision_extraction = tp_extraction / (tp_extraction + fp_extraction) if (
                                                                                                  tp_extraction + fp_extraction) > 0 else 0
            recall_extraction = tp_extraction / (tp_extraction + fn_extraction) if (
                                                                                               tp_extraction + fn_extraction) > 0 else 0
            f1_extraction = 2 * (precision_extraction * recall_extraction) / (
                        precision_extraction + recall_extraction) if (
                                                                                 precision_extraction + recall_extraction) > 0 else 0

            # 分类指标
            tp_classification = stats['category_correct']  # 正确分类的地址数
            fp_classification = max(0, stats['total_test'] - tp_classification)  # 错误分类的地址数
            fn_classification = max(0, stats['total_val'] - tp_classification)  # 漏分类的地址数

            precision_classification = tp_classification / (tp_classification + fp_classification) if (
                                                                                                                  tp_classification + fp_classification) > 0 else 0
            recall_classification = tp_classification / (tp_classification + fn_classification) if (
                                                                                                               tp_classification + fn_classification) > 0 else 0
            f1_classification = 2 * (precision_classification * recall_classification) / (
                        precision_classification + recall_classification) if (
                                                                                         precision_classification + recall_classification) > 0 else 0

            category_metrics[category] = {
                'extraction': {
                    'true_positive': tp_extraction,
                    'false_positive': fp_extraction,
                    'false_negative': fn_extraction,
                    'precision': precision_extraction,
                    'recall': recall_extraction,
                    'f1_score': f1_extraction
                },
                'classification': {
                    'true_positive': tp_classification,
                    'false_positive': fp_classification,
                    'false_negative': fn_classification,
                    'precision': precision_classification,
                    'recall': recall_classification,
                    'f1_score': f1_classification
                }
            }

    return category_metrics


def calculate_extraction_metrics(metrics: Dict) -> Dict:
    """计算地址提取的详细指标"""

    # 基础统计
    total_val_addresses = metrics['total_val_crimes']
    total_test_addresses = metrics['total_test_crimes']
    correctly_identified = metrics['address_correct']

    # 计算错误识别地址数
    incorrectly_identified = total_test_addresses - correctly_identified

    # 计算欠识别和过识别
    under_identified = max(0, total_val_addresses - correctly_identified)
    over_identified = max(0, total_test_addresses - correctly_identified)

    # 计算精度、召回率和F1分数
    precision = correctly_identified / total_test_addresses if total_test_addresses > 0 else 0
    recall = correctly_identified / total_val_addresses if total_val_addresses > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    extraction_metrics = {
        'total_val_addresses': total_val_addresses,
        'total_test_addresses': total_test_addresses,
        'correctly_identified_addresses': correctly_identified,
        'incorrectly_identified_addresses': incorrectly_identified,
        'under_identified_addresses': under_identified,
        'over_identified_addresses': over_identified,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }

    return extraction_metrics


def calculate_classification_metrics(metrics: Dict) -> Dict:
    """计算地址分类的详细指标"""

    total_val_addresses = metrics['total_val_crimes']
    correctly_classified = metrics['category_correct']

    # 计算精度、召回率和F1分数
    precision = correctly_classified / metrics['total_test_crimes'] if metrics['total_test_crimes'] > 0 else 0
    recall = correctly_classified / total_val_addresses if total_val_addresses > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    classification_metrics = {
        'total_val_addresses': total_val_addresses,
        'total_test_addresses': metrics['total_test_crimes'],
        'correctly_classified_addresses': correctly_classified,
        'incorrectly_classified_addresses': metrics['total_test_crimes'] - correctly_classified,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }

    return classification_metrics


def calculate_detailed_metrics(validation_file: str, test_result_file: str) -> Dict:
    """
    计算详细的评估指标
    """

    # 读取数据
    try:
        val_df = pd.read_csv(validation_file)
        test_df = pd.read_csv(test_result_file)
    except Exception as e:
        print(f"读取文件错误: {e}")
        return {}

    print(f"验证集样本数: {len(val_df)}")
    print(f"测试集样本数: {len(test_df)}")

    # 合并数据
    try:
        merged_df = pd.merge(val_df, test_df, on='UUID', how='inner',
                             suffixes=('_val', '_test'))
        print(f"合并后样本数: {len(merged_df)}")
    except Exception as e:
        print(f"数据合并错误: {e}")
        return {}

    # 初始化分类统计
    category_stats = defaultdict(lambda: {
        'total_val': 0,
        'total_test': 0,
        'address_correct': 0,
        'category_correct': 0,
        'samples': 0
    })

    metrics = {
        'total_samples': len(merged_df),
        'address_correct': 0,
        'category_correct': 0,
        'total_val_crimes': 0,
        'total_test_crimes': 0,
        'per_sample_metrics': [],
        'category_details': category_stats,
        'confusion_matrix': defaultdict(lambda: defaultdict(int))
    }

    processed_count = 0
    for idx, row in merged_df.iterrows():
        if processed_count % 100 == 0:
            print(f"处理进度: {processed_count}/{len(merged_df)}")
        processed_count += 1

        try:
            val_crimes = parse_crime_data(row['LLM'])
            test_crimes = parse_crime_data(row['Model'])
        except Exception as e:
            print(f"解析第 {idx} 行数据错误: {e}")
            continue

        sample_metrics = {
            'uuid': row['UUID'],
            'val_crime_count': len(val_crimes),
            'test_crime_count': len(test_crimes),
            'address_matches': 0,
            'category_matches': 0,
            'crime_details': []
        }

        metrics['total_val_crimes'] += len(val_crimes)
        metrics['total_test_crimes'] += len(test_crimes)

        # 更新分类统计 - 确保category是字符串
        for val_address, val_category in val_crimes:
            # 确保category是字符串
            category_key = safe_string_convert(val_category)
            if isinstance(category_key, list):
                category_key = str(category_key)
            metrics['category_details'][category_key]['total_val'] += 1
            metrics['category_details'][category_key]['samples'] += 1

        for test_address, test_category in test_crimes:
            category_key = safe_string_convert(test_category)
            if isinstance(category_key, list):
                category_key = str(category_key)
            metrics['category_details'][category_key]['total_test'] += 1

        # 匹配逻辑
        matched_test_indices = set()

        for i, (val_address, val_category) in enumerate(val_crimes):
            best_match_idx = -1
            best_similarity = 0
            best_test_category = ""
            best_test_address = ""

            for j, (test_address, test_category) in enumerate(test_crimes):
                if j in matched_test_indices:
                    continue

                sim = similarity(val_address, test_address)
                if sim > best_similarity:
                    best_similarity = sim
                    best_match_idx = j
                    best_test_category = test_category
                    best_test_address = test_address

            crime_detail = {
                'val_address': val_address,
                'val_category': val_category,
                'matched': False,
                'similarity': 0,
                'test_address': '',
                'test_category': '',
                'category_correct': False
            }

            if best_match_idx != -1 and best_similarity > 0.9:
                crime_detail['matched'] = True
                crime_detail['similarity'] = best_similarity
                crime_detail['test_address'] = best_test_address
                crime_detail['test_category'] = best_test_category

                sample_metrics['address_matches'] += 1
                matched_test_indices.add(best_match_idx)

                # 确保分类比较使用字符串
                val_cat_str = safe_string_convert(val_category)
                test_cat_str = safe_string_convert(best_test_category)

                # 检查分类是否正确
                if val_cat_str == test_cat_str:
                    sample_metrics['category_matches'] += 1
                    crime_detail['category_correct'] = True

                    # 更新分类统计
                    metrics['category_details'][val_cat_str]['category_correct'] += 1

                # 更新地址正确统计
                metrics['category_details'][val_cat_str]['address_correct'] += 1

                # 更新混淆矩阵
                metrics['confusion_matrix'][val_cat_str][test_cat_str] += 1

            sample_metrics['crime_details'].append(crime_detail)

        metrics['address_correct'] += sample_metrics['address_matches']
        metrics['category_correct'] += sample_metrics['category_matches']
        metrics['per_sample_metrics'].append(sample_metrics)

    # 计算最终精度
    metrics['address_precision'] = metrics['address_correct'] / metrics['total_val_crimes'] if metrics[
                                                                                                   'total_val_crimes'] > 0 else 0
    metrics['category_precision'] = metrics['category_correct'] / metrics['total_val_crimes'] if metrics[
                                                                                                     'total_val_crimes'] > 0 else 0

    # 计算每个分类的精度
    for category, stats in metrics['category_details'].items():
        if stats['total_val'] > 0:
            stats['address_precision'] = stats['address_correct'] / stats['total_val']
            stats['category_precision'] = stats['category_correct'] / stats['total_val']
        else:
            stats['address_precision'] = 0
            stats['category_precision'] = 0

    # 计算提取和分类的详细指标
    metrics['extraction_metrics'] = calculate_extraction_metrics(metrics)
    metrics['classification_metrics'] = calculate_classification_metrics(metrics)

    # 计算每个分类的详细指标
    metrics['category_metrics'] = calculate_category_metrics(metrics)

    print(f"\n处理完成!")
    print(f"总验证集犯罪记录: {metrics['total_val_crimes']}")
    print(f"总测试集犯罪记录: {metrics['total_test_crimes']}")

    return metrics


def save_results_to_file(metrics: Dict, validation_file: str, test_result_file: str, output_dir: str = "./results"):
    """将评估结果保存到文件"""

    # 从输入文件名生成输出文件名
    val_base = os.path.splitext(os.path.basename(validation_file))[0]
    test_base = os.path.splitext(os.path.basename(test_result_file))[0]
    base_output_name = f"{test_base}"

    # 主要结果文件
    main_output_file = os.path.join(output_dir, f"{base_output_name}_evaluation_results.csv")

    # 创建结果DataFrame
    results = []

    # 总体结果
    overall_row = {
        'Metric_Type': 'OVERALL',
        'Category': 'ALL',
        'Total_Samples': metrics['total_samples'],
        'Total_Val_Addresses': metrics['total_val_crimes'],
        'Total_Test_Addresses': metrics['total_test_crimes'],
        'Correct_Matches': metrics['address_correct'],
        'Precision': f"{metrics['address_precision']:.4f}",
        'Recall': f"{metrics['address_precision']:.4f}",  # 这里精度和召回相同
        'F1_Score': f"{metrics['address_precision']:.4f}"  # 简化计算
    }
    results.append(overall_row)

    # 地址提取详细指标
    ext_metrics = metrics['extraction_metrics']
    extraction_row = {
        'Metric_Type': 'ADDRESS_EXTRACTION',
        'Category': 'EXTRACTION_METRICS',
        'Total_Val_Addresses': ext_metrics['total_val_addresses'],
        'Total_Test_Addresses': ext_metrics['total_test_addresses'],
        'Correctly_Identified_Addresses': ext_metrics['correctly_identified_addresses'],
        'Incorrectly_Identified_Addresses': ext_metrics['incorrectly_identified_addresses'],
        'Under_identified_Addresses': ext_metrics['under_identified_addresses'],
        'Over_identified_Addresses': ext_metrics['over_identified_addresses'],
        'Precision': f"{ext_metrics['precision']:.4f}",
        'Recall': f"{ext_metrics['recall']:.4f}",
        'F1_Score': f"{ext_metrics['f1_score']:.4f}"
    }
    results.append(extraction_row)

    # 地址分类详细指标
    cls_metrics = metrics['classification_metrics']
    classification_row = {
        'Metric_Type': 'ADDRESS_CLASSIFICATION',
        'Category': 'CLASSIFICATION_METRICS',
        'Total_Val_Addresses': cls_metrics['total_val_addresses'],
        'Total_Test_Addresses': cls_metrics['total_test_addresses'],
        'Correctly_Classified_Addresses': cls_metrics['correctly_classified_addresses'],
        'Incorrectly_Classified_Addresses': cls_metrics['incorrectly_classified_addresses'],
        'Precision': f"{cls_metrics['precision']:.4f}",
        'Recall': f"{cls_metrics['recall']:.4f}",
        'F1_Score': f"{cls_metrics['f1_score']:.4f}"
    }
    results.append(classification_row)

    # 每个分类的详细结果（包含Precision, Recall, F1）
    for category, stats in metrics['category_details'].items():
        if stats['total_val'] > 0:  # 只显示有验证数据的分类
            cat_metrics = metrics['category_metrics'].get(category, {})
            ext_metrics_cat = cat_metrics.get('extraction', {})
            cls_metrics_cat = cat_metrics.get('classification', {})

            category_row = {
                'Metric_Type': 'PER_CATEGORY_DETAILED',
                'Category': category,
                'Total_Samples': stats['samples'],
                'Total_Val_Addresses': stats['total_val'],
                'Total_Test_Addresses': stats['total_test'],

                # 提取指标
                'Extraction_True_Positive': ext_metrics_cat.get('true_positive', 0),
                'Extraction_False_Positive': ext_metrics_cat.get('false_positive', 0),
                'Extraction_False_Negative': ext_metrics_cat.get('false_negative', 0),
                'Extraction_Precision': f"{ext_metrics_cat.get('precision', 0):.4f}",
                'Extraction_Recall': f"{ext_metrics_cat.get('recall', 0):.4f}",
                'Extraction_F1_Score': f"{ext_metrics_cat.get('f1_score', 0):.4f}",

                # 分类指标
                'Classification_True_Positive': cls_metrics_cat.get('true_positive', 0),
                'Classification_False_Positive': cls_metrics_cat.get('false_positive', 0),
                'Classification_False_Negative': cls_metrics_cat.get('false_negative', 0),
                'Classification_Precision': f"{cls_metrics_cat.get('precision', 0):.4f}",
                'Classification_Recall': f"{cls_metrics_cat.get('recall', 0):.4f}",
                'Classification_F1_Score': f"{cls_metrics_cat.get('f1_score', 0):.4f}"
            }
            results.append(category_row)

    # 保存到CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(main_output_file, index=False, encoding='utf-8-sig')

    # 保存分类详细指标到单独文件
    category_detailed_file = os.path.join(output_dir, f"{base_output_name}_category_detailed_metrics.csv")
    category_detailed_data = []

    for category, cat_metrics in metrics['category_metrics'].items():
        ext_metrics_cat = cat_metrics['extraction']
        cls_metrics_cat = cat_metrics['classification']

        row = {
            'Category': category,
            # 提取指标
            'Extraction_TP': ext_metrics_cat['true_positive'],
            'Extraction_FP': ext_metrics_cat['false_positive'],
            'Extraction_FN': ext_metrics_cat['false_negative'],
            'Extraction_Precision': f"{ext_metrics_cat['precision']:.4f}",
            'Extraction_Recall': f"{ext_metrics_cat['recall']:.4f}",
            'Extraction_F1_Score': f"{ext_metrics_cat['f1_score']:.4f}",
            # 分类指标
            'Classification_TP': cls_metrics_cat['true_positive'],
            'Classification_FP': cls_metrics_cat['false_positive'],
            'Classification_FN': cls_metrics_cat['false_negative'],
            'Classification_Precision': f"{cls_metrics_cat['precision']:.4f}",
            'Classification_Recall': f"{cls_metrics_cat['recall']:.4f}",
            'Classification_F1_Score': f"{cls_metrics_cat['f1_score']:.4f}"
        }
        category_detailed_data.append(row)

    category_detailed_df = pd.DataFrame(category_detailed_data)
    category_detailed_df.to_csv(category_detailed_file, index=False, encoding='utf-8-sig')

    # 保存混淆矩阵
    confusion_file = os.path.join(output_dir, f"{base_output_name}_confusion_matrix.csv")
    all_categories = sorted(set(metrics['confusion_matrix'].keys()) |
                            set(cat for preds in metrics['confusion_matrix'].values() for cat in preds.keys()))

    confusion_data = []
    # 表头
    header_row = {'Actual\\Predicted': ''}
    header_row.update({cat: cat for cat in all_categories})
    confusion_data.append(header_row)

    # 数据行
    for actual_cat in all_categories:
        row = {'Actual\\Predicted': actual_cat}
        for pred_cat in all_categories:
            row[pred_cat] = metrics['confusion_matrix'][actual_cat][pred_cat]
        confusion_data.append(row)

    confusion_df = pd.DataFrame(confusion_data)
    confusion_df.to_csv(confusion_file, index=False, encoding='utf-8-sig')

    # 保存详细样本结果
    detail_file = os.path.join(output_dir, f"{base_output_name}_detailed_samples.csv")
    detailed_samples = []

    for sample in metrics['per_sample_metrics']:
        sample_row = {
            'UUID': sample['uuid'],
            'Val_Crime_Count': sample['val_crime_count'],
            'Test_Crime_Count': sample['test_crime_count'],
            'Address_Matches': sample['address_matches'],
            'Category_Matches': sample['category_matches'],
            'Address_Precision': f"{(sample['address_matches'] / sample['val_crime_count']) if sample['val_crime_count'] > 0 else 0:.4f}",
            'Category_Precision': f"{(sample['category_matches'] / sample['val_crime_count']) if sample['val_crime_count'] > 0 else 0:.4f}"
        }
        detailed_samples.append(sample_row)

    detailed_df = pd.DataFrame(detailed_samples)
    detailed_df.to_csv(detail_file, index=False, encoding='utf-8-sig')

    return main_output_file, category_detailed_file, confusion_file, detail_file


def generate_summary_report(metrics: Dict, validation_file: str, test_result_file: str, output_dir: str = "."):
    """生成详细的文本报告"""

    val_base = os.path.splitext(os.path.basename(validation_file))[0]
    test_base = os.path.splitext(os.path.basename(test_result_file))[0]
    report_file = os.path.join(output_dir, f"{test_base}_evaluation_summary.txt")

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("地址提取与分类算法评估报告\n")
        f.write("=" * 80 + "\n\n")

        f.write("1. 总体评估结果\n")
        f.write("-" * 60 + "\n")
        f.write(f"总样本数: {metrics['total_samples']}\n")
        f.write(f"验证集地址总数: {metrics['total_val_crimes']}\n")
        f.write(f"测试集地址总数: {metrics['total_test_crimes']}\n")
        f.write(f"正确匹配的地址数: {metrics['address_correct']}\n")
        f.write(f"正确匹配的分类数: {metrics['category_correct']}\n")
        f.write(f"地址提取精度: {metrics['address_precision']:.4f} ({metrics['address_precision'] * 100:.2f}%)\n")
        f.write(f"地址分类精度: {metrics['category_precision']:.4f} ({metrics['category_precision'] * 100:.2f}%)\n\n")

        # 地址提取指标
        ext_metrics = metrics['extraction_metrics']
        f.write("2. 地址提取详细指标\n")
        f.write("-" * 60 + "\n")
        f.write(f"验证集地址总数: {ext_metrics['total_val_addresses']}\n")
        f.write(f"模型预测地址总数: {ext_metrics['total_test_addresses']}\n")
        f.write(f"正确识别地址数: {ext_metrics['correctly_identified_addresses']}\n")
        f.write(f"错误识别地址数: {ext_metrics['incorrectly_identified_addresses']}\n")
        f.write(f"欠识别地址数: {ext_metrics['under_identified_addresses']}\n")
        f.write(f"过识别地址数: {ext_metrics['over_identified_addresses']}\n")
        f.write(f"精度 (Precision): {ext_metrics['precision']:.4f} ({ext_metrics['precision'] * 100:.2f}%)\n")
        f.write(f"召回率 (Recall): {ext_metrics['recall']:.4f} ({ext_metrics['recall'] * 100:.2f}%)\n")
        f.write(f"F1分数: {ext_metrics['f1_score']:.4f} ({ext_metrics['f1_score'] * 100:.2f}%)\n\n")

        # 地址分类指标
        cls_metrics = metrics['classification_metrics']
        f.write("3. 地址分类详细指标\n")
        f.write("-" * 60 + "\n")
        f.write(f"验证集地址总数: {cls_metrics['total_val_addresses']}\n")
        f.write(f"模型预测地址总数: {cls_metrics['total_test_addresses']}\n")
        f.write(f"正确分类地址数: {cls_metrics['correctly_classified_addresses']}\n")
        f.write(f"错误分类地址数: {cls_metrics['incorrectly_classified_addresses']}\n")
        f.write(f"精度 (Precision): {cls_metrics['precision']:.4f} ({cls_metrics['precision'] * 100:.2f}%)\n")
        f.write(f"召回率 (Recall): {cls_metrics['recall']:.4f} ({cls_metrics['recall'] * 100:.2f}%)\n")
        f.write(f"F1分数: {cls_metrics['f1_score']:.4f} ({cls_metrics['f1_score'] * 100:.2f}%)\n\n")

        f.write("4. 按分类详细精度（Precision, Recall, F1）\n")
        f.write("-" * 60 + "\n")
        f.write("地址提取指标:\n")
        f.write(f"{'分类':<30} {'TP':<6} {'FP':<6} {'FN':<6} {'Precision':<12} {'Recall':<12} {'F1':<12}\n")
        f.write("-" * 90 + "\n")

        for category, cat_metrics in sorted(metrics['category_metrics'].items()):
            ext = cat_metrics['extraction']
            f.write(f"{category:<30} {ext['true_positive']:<6} {ext['false_positive']:<6} {ext['false_negative']:<6} "
                    f"{ext['precision']:<12.4f} {ext['recall']:<12.4f} {ext['f1_score']:<12.4f}\n")

        f.write("\n地址分类指标:\n")
        f.write(f"{'分类':<30} {'TP':<6} {'FP':<6} {'FN':<6} {'Precision':<12} {'Recall':<12} {'F1':<12}\n")
        f.write("-" * 90 + "\n")

        for category, cat_metrics in sorted(metrics['category_metrics'].items()):
            cls = cat_metrics['classification']
            f.write(f"{category:<30} {cls['true_positive']:<6} {cls['false_positive']:<6} {cls['false_negative']:<6} "
                    f"{cls['precision']:<12.4f} {cls['recall']:<12.4f} {cls['f1_score']:<12.4f}\n")


# 使用示例
if __name__ == "__main__":
    # 文件路径
    files = ['LEC_demo3_processed_data_json.csv', 'GLM4_processed_data_json.csv', 'qwen2.5_processed_data_json.csv', 'qwen3_processed_data_json.csv']
    validation_file = "./input/测试集/Test_Set_500_json.csv"  # 验证集文件路径
    test_result_file = "./output/" + files[2]   # 测试结果文件路径
    output_dir = "./output/report"  # 输出目录

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    try:
        # 计算详细指标
        print("正在计算评估指标...")
        metrics = calculate_detailed_metrics(validation_file, test_result_file)

        if not metrics:
            print("计算指标失败，请检查数据文件")
            exit(1)

        # 保存结果到文件
        print("正在保存结果...")
        main_result_file, category_detailed_file, confusion_file, detail_file = save_results_to_file(
            metrics, validation_file, test_result_file, output_dir
        )

        # 生成总结报告
        generate_summary_report(metrics, validation_file, test_result_file, output_dir)

        print("=" * 80)
        print("评估完成！")
        print("=" * 80)

        # 打印分类详细指标
        print("\n分类详细指标预览:")
        print("地址提取:")
        print(f"{'分类':<25} {'Precision':<10} {'Recall':<10} {'F1':<10}")
        print("-" * 55)
        for category, cat_metrics in sorted(metrics['category_metrics'].items()):
            ext = cat_metrics['extraction']
            print(f"{category:<25} {ext['precision']:<10.4f} {ext['recall']:<10.4f} {ext['f1_score']:<10.4f}")

        print(f"\n生成的文件:")
        print(f"主要结果: {main_result_file}")
        print(f"分类详细指标: {category_detailed_file}")
        print(f"混淆矩阵: {confusion_file}")
        print(f"详细样本结果: {detail_file}")
        print(
            f"总结报告: {os.path.join(output_dir, os.path.basename(test_result_file).replace('.csv', '') + '_evaluation_summary.txt')}")
    except Exception as e:
        print(f"程序执行错误: {e}")
        import traceback

        traceback.print_exc()