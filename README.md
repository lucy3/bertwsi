### Lucy's fork of Amrami & Goldberg 2019

Original paper: Towards better substitution-based word sense induction - https://arxiv.org/abs/1905.12598

Adapted for Reddit data: https://github.com/lucy3/ingroup_lang

### Prerequisites:
Python 3.7<br>
Install requirements.txt with pip -r<br>
This will install python pacakges including pytorch and huggingface's BERT port.<br>
(for CUDA support first install pytorch accroding to [their instructions](https://pytorch.org/)).<br>

Run download_resources.sh to download datasets.

### WSI:
Run wsi_bert.py for sense induction on both SemEval 2010 and 2013 WSI task datasets. <br>
Logs should be printed to "debug" dir. 

### Additions by Lucy

- train\_semeval.py: training for SemEval tasks
- reddit\_prep.py: formats Reddit data
- wsi\_reddit.py: train / learn clusters on Reddit examples
- match\_reddit.py: matches Reddit examples to clusters obtained after training
- inspect\_senses.py: examines resulting senses
- scheduler.py: for submitting jobs
- wsi/wsi\_clustering.py: includes modifications that save learned clusters and a matching process for new examples
