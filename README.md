# A-Novel-Patent-Similarity-Measurement-Methodology

Document Similarity Prediction

  * [A-Novel-Patent-Similarity-Measurement-Methodology: Semantic Distance and Technological Distance](주소추가)
  
    * `Yongmin Yoo`, `Cheonkam Jeong`, `Sanguk Gim`, `Junwon Lee`, `Deaho Seo`

-------------------------------------------------

## Dataset

  * We used Google patent, randomly extract 420 pairs patents from 2019 to 2020, with which we perform expert validation
  * The expert panel whom has expertise in Data Analytics, Data Mining, and Artificial Intelligence assesses how semantically similar two patents
  * We used the score of law expert when there was a large difference in scores between experts.
  
  |    Expert    | Frequency  |
  | :------: | :---: |
  |    Expert panel    | 335  |
  |    Law Expert    | 85  |
  |    Total    | 420  |
  
  
-------------------------------------------------

## Model Structure

  * Semantic Similarity
    
    * We used [Patent Bert](https://huggingface.co/anferico/bert-for-patents)

  * Technical Similarity
    
    * $Intersection_{A,B}=Patent_{A} \cap Patent_{B}$
    * $Union_{A,B}=Patent_{A} \cup Patent_{B}$
    * $TD_{A,B}={Intersection_{A,B} \over Union_{A,B}}$

  * Hybrid Similarity

    * $SDTD_{A,B}={(TD+1) \cdot SD \over 2}$
  
 -------------------------------------------------
 
 ## Result
 
  |    Model    | Pearson  | Spearman  |
  | :------: | :---: | :-----: |
  |  CNN              |  0.074951      | 0.035229      |
  |  LSTM              | 0.05409      | 0.05967      |
  |  Bi-LSTM            | 0.14646      | 0.185534      |
  |  BERT         | 0.507742      | 0.542219      |
  |  **Proposed model** | **0.56935**  | **0.644496**  |
