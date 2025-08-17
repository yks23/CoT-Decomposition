# from modelscope.msdatasets import MsDataset
# ds =  MsDataset.load('open-r1/DAPO-Math-17k-Processed', subset_name='all', split='train')
# # dapo
# # from modelscope.msdatasets import MsDataset
# ds =  MsDataset.load('opencompass/AIME2025', subset_name='AIME2025-I', split='test')
# # AIME 2025 or AIME2025-II
# # from modelscope.msdatasets import MsDataset
# ds =  MsDataset.load('AI-ModelScope/Maxwell-Jia-AIME_2024', subset_name='default', split='train')
# # AIME 2024 
# # from modelscope.msdatasets import MsDataset
# ds =  MsDataset.load('knoveleng/AMC-23', subset_name='default', split='train')

# # from modelscope.msdatasets import MsDataset
# ds =  MsDataset.load('AI-ModelScope/MATH-500', subset_name='default', split='test')

# # from modelscope.msdatasets import MsDataset
# ds =  MsDataset.load('knoveleng/Minerva-Math', subset_name='default', split='train')

# # from modelscope.msdatasets import MsDataset
# ds =  MsDataset.load('AI-ModelScope/OlympiadBench')
from datasets import load_dataset

ds = load_dataset("HuggingFaceH4/aime_2024")
print(ds['train'][0])