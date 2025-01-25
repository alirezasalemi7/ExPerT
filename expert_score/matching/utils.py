def _calculate_scores_for_a_macth_table(match_table):
    content_only_recall, content_only_precision = 0, 0
    stye_only_recall, style_only_precision = 0, 0
    content_and_style_recall, content_and_style_precision = 0, 0
    content_or_style_recall, content_or_style_precision = 0, 0
    content_style_average_recall, content_style_average_precision = 0, 0
    len_recall, len_precision = len(match_table['recall']), len(match_table['precision'])
    for aspect, match_details in match_table['recall'].items():
        if match_details['matched']:
            match_details = match_details['details']
            if match_details['is_content_matched']:
                content_only_recall += (1 / len_recall)
                content_style_average_recall += (0.5 / len_recall)
            if match_details['is_style_matched']:
                stye_only_recall += (1 / len_recall)
                content_style_average_recall += (0.5 / len_recall)
            if match_details['is_content_matched'] and match_details['is_style_matched']:
                content_and_style_recall += (1 / len_recall)
            if match_details['is_content_matched'] or match_details['is_style_matched']:
                content_or_style_recall += (1 / len_recall)
    for aspect, match_details in match_table['precision'].items():
        if match_details['matched']:
            match_details = match_details['details']
            if match_details['is_content_matched']:
                content_only_precision += (1 / len_precision)
                content_style_average_precision += (0.5 / len_precision)
            if match_details['is_style_matched']:
                style_only_precision += (1 / len_precision)
                content_style_average_precision += (0.5 / len_precision)
            if match_details['is_content_matched'] and match_details['is_style_matched']:
                content_and_style_precision += (1 / len_precision)
            if match_details['is_content_matched'] or match_details['is_style_matched']:
                content_or_style_precision += (1 / len_precision)
    content_only_recall = content_only_recall
    content_only_precision = content_only_precision
    stye_only_recall = stye_only_recall
    style_only_precision = style_only_precision
    content_and_style_recall = content_and_style_recall
    content_and_style_precision = content_and_style_precision
    content_or_style_recall = content_or_style_recall
    content_or_style_precision = content_or_style_precision
    content_style_average_recall = content_style_average_recall
    content_style_average_precision = content_style_average_precision
    content_only_f1 = 0 if content_only_precision + content_only_recall == 0 else 2 * content_only_precision * content_only_recall / (content_only_precision + content_only_recall)
    style_only_f1 = 0 if style_only_precision + stye_only_recall == 0 else 2 * style_only_precision * stye_only_recall / (style_only_precision + stye_only_recall)
    content_and_style_f1 = 0 if content_and_style_precision + content_and_style_recall == 0 else 2 * content_and_style_precision * content_and_style_recall / (content_and_style_precision + content_and_style_recall)
    content_or_style_f1 = 0 if content_or_style_precision + content_or_style_recall == 0 else 2 * content_or_style_precision * content_or_style_recall / (content_or_style_precision + content_or_style_recall)
    content_style_average_f1 = 0 if content_style_average_precision + content_style_average_recall == 0 else 2 * content_style_average_precision * content_style_average_recall / (content_style_average_precision + content_style_average_recall)
    return {
        "content_only_recall" : content_only_recall,
        "content_only_precision" : content_only_precision,
        "content_only_f1" : content_only_f1,
        "style_only_recall" : stye_only_recall,
        "style_only_precision" : style_only_precision,
        "style_only_f1" : style_only_f1,
        "content_and_style_recall" : content_and_style_recall,
        "content_and_style_precision" : content_and_style_precision,
        "content_and_style_f1" : content_and_style_f1,
        "content_or_style_recall" : content_or_style_recall,
        "content_or_style_precision" : content_or_style_precision,
        "content_or_style_f1" : content_or_style_f1,
        "content_style_average_recall" : content_style_average_recall,
        "content_style_average_precision" : content_style_average_precision,
        "content_style_average_f1" : content_style_average_f1
    }

def calculate_scores_for_batch(match_tables):
    results = {
        "per_query": {},
        "average": {}
    }
    for id, match_table in match_tables.items():
        results["per_query"][id] = _calculate_scores_for_a_macth_table(match_table)
    for id, scores in results['per_query'].items():
        for key, value in scores.items():
            if key not in results['average']:
                results['average'][key] = 0
            results['average'][key] += value
    for key in results['average']:
        results['average'][key] /= len(results['per_query'])
    return results