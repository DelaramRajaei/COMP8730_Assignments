# Introduction
This project assesses the effectiveness of the Minimum Edit Distance (MED) algorithm, utilizing WordNet as a lexical resource, in correcting misspelled tokens within the Birkbeck spelling error corpus. The MED algorithm, crucial in computational linguistics, gauges string similarity by determining the minimum single-character edits needed for transformation. WordNet serves as a valuable resource for evaluating proximity to potential corrections. The Birkbeck spelling error corpus, a real-world collection of misspellings, facilitates a comprehensive assessment of MED's efficacy. Calculating the average success at k (s@k) using Pytrec eval for all misspelled tokens provides a quantitative measure, offering insights into MED's practical applicability in English spelling correction.

# Setup and Installation
To work locally, you can clone this repository as follows.

```
git clone https://github.com/DelaramRajaei/COMP8730_Assignments.git
cd Assignment01
```
You can install the requirements using one of the following commands:
```
conda env create -f environment.yml
conda activate assign01
```
```
pip install -r requirement.txt
```
Then you can simply run the project using the following command:
```
python assingn01.py
```

As a quick starter pack, a [notebook](https://colab.research.google.com/drive/1NyCn4j8OtPQsmAMty41PN6Na9l6T9q4t) is considered with the running example and parameters.
