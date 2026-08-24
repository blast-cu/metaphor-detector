from src.utils import load_dotenv, load_json, save_to_json

def fix_ids(path, filename):

    data_w_config = load_json(path + filename)
    data = data_w_config["data"]
    final_data = {}
    for id, v in data.items():

        doc_id = id.split("_")[0]
        clean_doc_id = "-".join(doc_id.split("-")[0:-1])
        clean_met_id = ("_").join([clean_doc_id] + id.split("_")[1:])
        final_data[clean_met_id] = v

    data_w_config["data"] = final_data
    out_filename = filename.replace(".json", "-fixed.json")
    save_to_json(data_w_config, path, out_filename)


def main():

    fix_ids("results/feature_extractor/llm/", "podcasts_target_ppto_class_qwen.json")
    fix_ids("results/feature_extractor/llm/", "podcasts_target_person_class_qwen.json")
    fix_ids("results/feature_extractor/llm/", "podcasts_target_place_class_qwen.json")
    fix_ids("results/feature_extractor/llm/", "podcasts_target_thing_class_qwen.json")
    fix_ids("results/feature_extractor/llm/", "podcasts_target_organization_class_qwen.json")

    fix_ids("results/metaphor_classification/llm/", "podcasts_source_verb_target_noun_binary_met_class_qwen.json")
    fix_ids("results/gen_entailments/llm/", "podcasts_metaphor_framing_entailments_qwen.json")




if __name__ == main():
	main()
