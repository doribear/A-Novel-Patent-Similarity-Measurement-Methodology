# A Novel patent similarity measurement methodology

Document Similarity Prediction

  * [A-Novel-Patent-Similarity-Measurement-Methodology: Semantic Distance and Technological Distance](주소추가)
  
    * `Yongmin Yoo`, `Cheonkam Jeong`, `Sanguk Gim`, `Junwon Lee`, `Deaho Seo`

-------------------------------------------------

## Dataset

  * 추가예정

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

    추가예정
