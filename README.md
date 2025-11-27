# Vigilante
EEG project Team HART '25

1. Download the Data <br>
⚠️ Note: The dataset is not included in this repository due to licensing and file size limits.

Go to our team's [Notion Page](https://www.notion.so/teamhart/Vigilante-2b2096c92b6e80179cbaf98ae466af2f) and download drozy_data.zip.

Unzip the file inside this project folder.

Ensure your folder structure looks exactly like this:

```
Vigilante/
├── src/
├── models/
└── :INSERT_DATA_FOLDER:/              <-- Move the drozy_data folder under the ":INSERT_DATA_FOLDER:" folder 
    └── drozy_data/
        ├── KSS.txt
        ├── 1-1.edf
        ├── 1-2.edf
        └── ... (rest of the files)
```


2. Run the Training
Once the data is in place, run the training script to build the model
