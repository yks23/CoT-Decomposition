from datasets import Dataset

# ======================
# 数据层
# ======================

max_token_dataset = {
    "gsm8k": 2000,
    "math": 1500,
    "aime24": 8192,
    "aime25": 8192,
    "amc23": 2000,
    "math500": 3000,
    "minerva": 2000,
    "olympiad_bench": 4000,
    "openrl": 1000,
    "dapo": 3000,
    "medqa":1500,
    "medmcqa":1500,
    "pubmedqa":1500,
    "clinical_knowledge":1500,
    "college_biology":1500,
    "college_medicine":1500,
    "medical_genetics":1500,
    "professional_medicine":1500,
    "anatomy":1500,
    "entropy_bench": 3000,
}
is_multi_choice ={
    'medqa',
    'medmcqa'
    'pubmedqa','clinical_knowledge','college_biology','college_medicine','medical_genetics','professional_medicine','anatomy'
}
max_batch_size ={
    "gsm8k": 15,
    "math": 20,
    "aime24": 5,
    "aime25": 5,
    "amc23": 15,
    "math500": 10,
    "minerva": 15,
    "olympiad_bench": 8,
    "openrl": 20,
    "dapo": 5,
    "medqa":15,
    "medmcqa":15,
    "pubmedqa":15,
    "clinical_knowledge":15,
    "college_biology":15,
    "college_medicine":15,
    "medical_genetics":15,
    "professional_medicine":15,
    "anatomy":15,
    "entropy_bench": 10,
}

def load_dataset_by_name(name: str):
    mapping = {
        "gsm8k": "/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/gsm8k/test-new.parquet",
        "math": "/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/math-algebra/test-new.parquet",
        "aime24": "/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/aime24/raw.parquet",
        "aime25": "/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/aime25/default.parquet",
        "amc23": "/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/amc23/default.parquet",
        "math500": "/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/math500/default-new.parquet",
        "minerva": "/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/minerva/default.parquet",
        "olympiad_bench": "/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/olympiad_bench/default-new.parquet",
        "openrl": "/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/openrl/test.parquet",
        "openrl-train": "/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/openrl/sub-train.parquet",
        "openrl-raw-test": "/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/openrl/raw_test.parquet",
        "dapo": "/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/dapo/eval.parquet",
        "medqa": "/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/medqa/en.parquet",
        "medmcqa": "/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/medmcqa/processed.parquet",
        "pubmedqa": "/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/pubmedqa/processed.parquet",
        "clinical_knowledge": "/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/mmlu/clinical_knowledge.parquet",
        "college_biology": "/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/mmlu/college_biology.parquet",
        "college_medicine": "/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/mmlu/college_medicine.parquet",
        "medical_genetics": "/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/mmlu/medical_genetics.parquet",
        "professional_medicine": "/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/mmlu/professional_medicine.parquet",
        "anatomy": "/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/mmlu/anatomy.parquet",
        "entropy_bench": "/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/entropy_bench/raw.parquet",
        
    }
    if name.endswith(".parquet"):
        return Dataset.from_parquet(name), name.split("/")[-1].split(".")[0]
    return Dataset.from_parquet(mapping[name]), name
