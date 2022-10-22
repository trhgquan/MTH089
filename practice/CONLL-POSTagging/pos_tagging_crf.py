{"metadata":{"kernelspec":{"language":"python","display_name":"Python 3","name":"python3"},"language_info":{"name":"python","version":"3.7.12","mimetype":"text/x-python","codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3","nbconvert_exporter":"python","file_extension":".py"}},"nbformat_minor":4,"nbformat":4,"cells":[{"cell_type":"code","source":"# %% [code] {\"execution\":{\"iopub.status.busy\":\"2022-10-22T12:26:02.133894Z\",\"iopub.execute_input\":\"2022-10-22T12:26:02.134575Z\",\"iopub.status.idle\":\"2022-10-22T12:26:02.161572Z\",\"shell.execute_reply.started\":\"2022-10-22T12:26:02.134467Z\",\"shell.execute_reply\":\"2022-10-22T12:26:02.160401Z\"}}\n# This Python 3 environment comes with many helpful analytics libraries installed\n# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python\n# For example, here's several helpful packages to load\n\nimport numpy as np # linear algebra\nimport pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)\n\n# Input data files are available in the read-only \"../input/\" directory\n# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory\n\nimport os\nfor dirname, _, filenames in os.walk('/kaggle/input'):\n    for filename in filenames:\n        print(os.path.join(dirname, filename))\n\n# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using \"Save & Run All\" \n# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session\n\n# %% [markdown]\n# # Download and preprocess data\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2022-10-22T12:26:02.163275Z\",\"iopub.execute_input\":\"2022-10-22T12:26:02.163628Z\",\"iopub.status.idle\":\"2022-10-22T12:26:06.322882Z\",\"shell.execute_reply.started\":\"2022-10-22T12:26:02.163597Z\",\"shell.execute_reply\":\"2022-10-22T12:26:06.321659Z\"}}\n!curl -LJ \"https://raw.githubusercontent.com/ningshixian/NER-CONLL2003/master/data/train.txt\" -o \"train.txt\"\n!curl -LJ \"https://raw.githubusercontent.com/ningshixian/NER-CONLL2003/master/data/valid.txt\" -o \"valid.txt\"\n!curl -LJ \"https://raw.githubusercontent.com/ningshixian/NER-CONLL2003/master/data/test.txt\" -o \"test.txt\"\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2022-10-22T12:26:06.326186Z\",\"iopub.execute_input\":\"2022-10-22T12:26:06.326732Z\",\"iopub.status.idle\":\"2022-10-22T12:26:07.378357Z\",\"shell.execute_reply.started\":\"2022-10-22T12:26:06.326680Z\",\"shell.execute_reply\":\"2022-10-22T12:26:07.377182Z\"}}\n!head -5 train.txt\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2022-10-22T12:26:07.380177Z\",\"iopub.execute_input\":\"2022-10-22T12:26:07.380970Z\",\"iopub.status.idle\":\"2022-10-22T12:26:08.529196Z\",\"shell.execute_reply.started\":\"2022-10-22T12:26:07.380884Z\",\"shell.execute_reply\":\"2022-10-22T12:26:08.527825Z\"}}\nimport nltk\nfrom nltk.corpus.reader import ConllCorpusReader\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2022-10-22T12:26:08.531833Z\",\"iopub.execute_input\":\"2022-10-22T12:26:08.532231Z\",\"iopub.status.idle\":\"2022-10-22T12:26:08.540864Z\",\"shell.execute_reply.started\":\"2022-10-22T12:26:08.532197Z\",\"shell.execute_reply\":\"2022-10-22T12:26:08.539750Z\"}}\ntrain_sentences = ConllCorpusReader(\"./\", \"train.txt\", [\"words\", \"pos\", \"ignore\", \"chunk\"]).iob_sents()\nvalid_sentences = ConllCorpusReader(\"./\", \"valid.txt\", [\"words\", \"pos\", \"ignore\", \"chunk\"]).iob_sents()\n\n# %% [markdown]\n# Remove empty (len = 0) sentences due to data error.\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2022-10-22T12:26:08.542414Z\",\"iopub.execute_input\":\"2022-10-22T12:26:08.542742Z\",\"iopub.status.idle\":\"2022-10-22T12:26:10.008021Z\",\"shell.execute_reply.started\":\"2022-10-22T12:26:08.542713Z\",\"shell.execute_reply\":\"2022-10-22T12:26:10.006963Z\"}}\ntrain_sentences = [s for s in train_sentences if len(s) > 0]\nvalid_sentences = [s for s in valid_sentences if len(s) > 0]\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2022-10-22T12:26:10.009478Z\",\"iopub.execute_input\":\"2022-10-22T12:26:10.009916Z\",\"iopub.status.idle\":\"2022-10-22T12:26:10.021883Z\",\"shell.execute_reply.started\":\"2022-10-22T12:26:10.009875Z\",\"shell.execute_reply\":\"2022-10-22T12:26:10.020774Z\"}}\ntrain_sentences[0]\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2022-10-22T12:26:10.023286Z\",\"iopub.execute_input\":\"2022-10-22T12:26:10.024106Z\",\"iopub.status.idle\":\"2022-10-22T12:26:10.034489Z\",\"shell.execute_reply.started\":\"2022-10-22T12:26:10.024061Z\",\"shell.execute_reply\":\"2022-10-22T12:26:10.033501Z\"}}\nvalid_sentences[0]\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2022-10-22T12:26:10.035792Z\",\"iopub.execute_input\":\"2022-10-22T12:26:10.036577Z\",\"iopub.status.idle\":\"2022-10-22T12:26:10.045306Z\",\"shell.execute_reply.started\":\"2022-10-22T12:26:10.036532Z\",\"shell.execute_reply\":\"2022-10-22T12:26:10.044494Z\"}}\nprint(f\"Length of training set: {len(train_sentences)}\")\nprint(f\"Length of validation set: {len(valid_sentences)}\")\n\n# %% [markdown]\n# Making a Pandas dataframe, instead of list of tuples\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2022-10-22T12:26:10.046476Z\",\"iopub.execute_input\":\"2022-10-22T12:26:10.047229Z\",\"iopub.status.idle\":\"2022-10-22T12:26:10.371375Z\",\"shell.execute_reply.started\":\"2022-10-22T12:26:10.047195Z\",\"shell.execute_reply\":\"2022-10-22T12:26:10.369875Z\"}}\nframe = []\n\nfor s in train_sentences:\n    for term in s:\n        frame.append({\n            \"token\": term[0],\n            \"postag\": term[1],\n            \"label\": term[2]\n        })\n    frame.append({\n        \"token\": \"\",\n        \"postag\": \"\",\n        \"label\": \"\"\n    })\n\ndf = pd.DataFrame(frame)\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2022-10-22T12:26:10.375219Z\",\"iopub.execute_input\":\"2022-10-22T12:26:10.375582Z\",\"iopub.status.idle\":\"2022-10-22T12:26:10.381202Z\",\"shell.execute_reply.started\":\"2022-10-22T12:26:10.375551Z\",\"shell.execute_reply\":\"2022-10-22T12:26:10.380124Z\"}}\npd.set_option(\"display.max_column\", None)\npd.set_option(\"max_rows\", 10)\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2022-10-22T12:26:10.382734Z\",\"iopub.execute_input\":\"2022-10-22T12:26:10.383148Z\",\"iopub.status.idle\":\"2022-10-22T12:26:10.409113Z\",\"shell.execute_reply.started\":\"2022-10-22T12:26:10.383104Z\",\"shell.execute_reply\":\"2022-10-22T12:26:10.408046Z\"}}\ndf\n\n# %% [markdown]\n# # Feature extraction\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2022-10-22T12:26:10.410475Z\",\"iopub.execute_input\":\"2022-10-22T12:26:10.410825Z\",\"iopub.status.idle\":\"2022-10-22T12:26:10.421889Z\",\"shell.execute_reply.started\":\"2022-10-22T12:26:10.410794Z\",\"shell.execute_reply\":\"2022-10-22T12:26:10.420591Z\"}}\ndef word2feat(sentence, idx):\n    word = sentence[idx][0]\n    postag = sentence[idx][1]\n    \n    features = {\n        \"bias\" : 1.0,\n        \"word.lower()\" : word.lower(),\n        \"word[-3:]\" : word[-3:],\n        \"word[-2:]\" : word[-2:],\n        \"word.isupper()\" : word.isupper(),\n        \"word.istitle()\" : word.istitle(),\n        \"word.isdigit()\" : word.isdigit(),\n        \"postag\" : postag,\n        \"postag[:2]\" : postag[:2]\n    }\n    \n    if idx > 0:\n        # This word is not standing at the sentence's beginning\n        word1 = sentence[idx - 1][0]\n        postag1 = sentence[idx - 1][1]\n        \n        features.update({\n            \"-1:word.lower()\" : word1.lower(),\n            \"-1:word.istitle()\" : word1.istitle(),\n            \"-1:word.isupper()\" : word1.isupper(),\n            \"-1:postag\" : postag1,\n            \"-1:postag[:2]\" : postag1[:2]\n        })\n    else:\n        features[\"BOS\"] = True\n    \n    if idx < len(sentence) - 1:\n        # This word is not standing at the sentence's end\n        word1 = sentence[idx + 1][0]\n        postag1 = sentence[idx + 1][1]\n        \n        features.update({\n            \"+1:word.lower()\" : word1.lower(),\n            \"+1:word.istitle()\" : word1.istitle(),\n            \"+1:word.isupper()\" : word1.isupper(),\n            \"+1:postag\" : postag1,\n            \"+1:postag[:2]\" : postag1[:2]\n        })\n    else:\n        features[\"EOS\"] = True\n    \n    return features    \n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2022-10-22T12:26:10.423337Z\",\"iopub.execute_input\":\"2022-10-22T12:26:10.423693Z\",\"iopub.status.idle\":\"2022-10-22T12:26:10.435532Z\",\"shell.execute_reply.started\":\"2022-10-22T12:26:10.423661Z\",\"shell.execute_reply\":\"2022-10-22T12:26:10.434115Z\"}}\ndef sent2feat(sentence):\n    return [word2feat(sentence, i) for i in range(len(sentence))]\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2022-10-22T12:26:10.437221Z\",\"iopub.execute_input\":\"2022-10-22T12:26:10.437971Z\",\"iopub.status.idle\":\"2022-10-22T12:26:10.447274Z\",\"shell.execute_reply.started\":\"2022-10-22T12:26:10.437911Z\",\"shell.execute_reply\":\"2022-10-22T12:26:10.446317Z\"}}\ndef sent2labels(sentence):\n    return [label for _, _, label in sentence]\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2022-10-22T12:26:10.448691Z\",\"iopub.execute_input\":\"2022-10-22T12:26:10.449452Z\",\"iopub.status.idle\":\"2022-10-22T12:26:10.458773Z\",\"shell.execute_reply.started\":\"2022-10-22T12:26:10.449406Z\",\"shell.execute_reply\":\"2022-10-22T12:26:10.457878Z\"}}\ndef sent2tokens(sentence):\n    return [token for token, _, _ in sentence]\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2022-10-22T12:26:10.460194Z\",\"iopub.execute_input\":\"2022-10-22T12:26:10.461012Z\",\"iopub.status.idle\":\"2022-10-22T12:26:10.482289Z\",\"shell.execute_reply.started\":\"2022-10-22T12:26:10.460973Z\",\"shell.execute_reply\":\"2022-10-22T12:26:10.481312Z\"}}\nsent2feat(train_sentences[0])\n\n# %% [markdown]\n# Make CRF training and valid data \n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2022-10-22T12:26:10.483458Z\",\"iopub.execute_input\":\"2022-10-22T12:26:10.484312Z\",\"iopub.status.idle\":\"2022-10-22T12:26:11.539277Z\",\"shell.execute_reply.started\":\"2022-10-22T12:26:10.484275Z\",\"shell.execute_reply\":\"2022-10-22T12:26:11.538138Z\"}}\nX_train = [sent2feat(s) for s in train_sentences]\ny_train = [sent2labels(s) for s in train_sentences]\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2022-10-22T12:26:11.540484Z\",\"iopub.execute_input\":\"2022-10-22T12:26:11.540805Z\",\"iopub.status.idle\":\"2022-10-22T12:26:11.778365Z\",\"shell.execute_reply.started\":\"2022-10-22T12:26:11.540776Z\",\"shell.execute_reply\":\"2022-10-22T12:26:11.777096Z\"}}\nX_valid = [sent2feat(s) for s in valid_sentences]\ny_valid = [sent2labels(s) for s in valid_sentences]\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2022-10-22T12:26:11.779692Z\",\"iopub.execute_input\":\"2022-10-22T12:26:11.780044Z\",\"iopub.status.idle\":\"2022-10-22T12:26:11.786635Z\",\"shell.execute_reply.started\":\"2022-10-22T12:26:11.780011Z\",\"shell.execute_reply\":\"2022-10-22T12:26:11.785365Z\"}}\nprint(len(X_train[0]))\nprint(len(y_train[0]))\nprint(X_train[0][2])\nprint(y_train[0][2])\n\n# %% [markdown]\n# # Training\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2022-10-22T12:26:11.788055Z\",\"iopub.execute_input\":\"2022-10-22T12:26:11.788469Z\",\"iopub.status.idle\":\"2022-10-22T12:26:27.183757Z\",\"shell.execute_reply.started\":\"2022-10-22T12:26:11.788436Z\",\"shell.execute_reply\":\"2022-10-22T12:26:27.182040Z\"}}\n!pip install git+https://github.com/MeMartijn/updated-sklearn-crfsuite.git#egg=sklearn_crfsuite\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2022-10-22T12:26:27.185800Z\",\"iopub.execute_input\":\"2022-10-22T12:26:27.186305Z\",\"iopub.status.idle\":\"2022-10-22T12:26:27.208836Z\",\"shell.execute_reply.started\":\"2022-10-22T12:26:27.186252Z\",\"shell.execute_reply\":\"2022-10-22T12:26:27.207515Z\"}}\nimport sklearn_crfsuite\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2022-10-22T12:26:27.211135Z\",\"iopub.execute_input\":\"2022-10-22T12:26:27.211858Z\",\"iopub.status.idle\":\"2022-10-22T12:26:27.217449Z\",\"shell.execute_reply.started\":\"2022-10-22T12:26:27.211810Z\",\"shell.execute_reply\":\"2022-10-22T12:26:27.216328Z\"}}\ncrf = sklearn_crfsuite.CRF(\n    algorithm = \"lbfgs\",\n    c1 = 0.1,\n    c2 = 0.1,\n    max_iterations = 100,\n    all_possible_transitions = True,\n    verbose = True\n)\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2022-10-22T12:26:27.218926Z\",\"iopub.execute_input\":\"2022-10-22T12:26:27.219372Z\",\"iopub.status.idle\":\"2022-10-22T12:27:18.460763Z\",\"shell.execute_reply.started\":\"2022-10-22T12:26:27.219329Z\",\"shell.execute_reply\":\"2022-10-22T12:27:18.459477Z\"}}\ncrf.fit(X_train, y_train)\n\n# %% [markdown]\n# # Evaluation\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2022-10-22T12:27:18.462457Z\",\"iopub.execute_input\":\"2022-10-22T12:27:18.462942Z\",\"iopub.status.idle\":\"2022-10-22T12:27:18.469951Z\",\"shell.execute_reply.started\":\"2022-10-22T12:27:18.462871Z\",\"shell.execute_reply\":\"2022-10-22T12:27:18.468755Z\"}}\nfrom sklearn_crfsuite import metrics\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2022-10-22T12:27:18.471368Z\",\"iopub.execute_input\":\"2022-10-22T12:27:18.471736Z\",\"iopub.status.idle\":\"2022-10-22T12:27:18.495201Z\",\"shell.execute_reply.started\":\"2022-10-22T12:27:18.471704Z\",\"shell.execute_reply\":\"2022-10-22T12:27:18.493891Z\"}}\nlabels = list(crf.classes_)\nprint(labels)\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2022-10-22T12:27:18.497201Z\",\"iopub.execute_input\":\"2022-10-22T12:27:18.497887Z\",\"iopub.status.idle\":\"2022-10-22T12:27:19.188777Z\",\"shell.execute_reply.started\":\"2022-10-22T12:27:18.497805Z\",\"shell.execute_reply\":\"2022-10-22T12:27:19.187558Z\"}}\ny_pred = crf.predict(X_valid)\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2022-10-22T12:27:19.190303Z\",\"iopub.execute_input\":\"2022-10-22T12:27:19.190669Z\",\"iopub.status.idle\":\"2022-10-22T12:27:19.430445Z\",\"shell.execute_reply.started\":\"2022-10-22T12:27:19.190635Z\",\"shell.execute_reply\":\"2022-10-22T12:27:19.429263Z\"}}\nprint(metrics.flat_f1_score(y_valid, y_pred,\n                      average='macro', labels=labels))\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2022-10-22T12:27:19.436709Z\",\"iopub.execute_input\":\"2022-10-22T12:27:19.437068Z\",\"iopub.status.idle\":\"2022-10-22T12:27:19.678235Z\",\"shell.execute_reply.started\":\"2022-10-22T12:27:19.437036Z\",\"shell.execute_reply\":\"2022-10-22T12:27:19.676805Z\"}}\nprint(metrics.flat_precision_score(y_valid, y_pred,\n                      average='macro', labels=labels))\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2022-10-22T12:27:19.679617Z\",\"iopub.execute_input\":\"2022-10-22T12:27:19.680005Z\",\"iopub.status.idle\":\"2022-10-22T12:27:19.918238Z\",\"shell.execute_reply.started\":\"2022-10-22T12:27:19.679971Z\",\"shell.execute_reply\":\"2022-10-22T12:27:19.916922Z\"}}\nprint(metrics.flat_recall_score(y_valid, y_pred,\n                      average='macro', labels=labels))\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2022-10-22T12:27:19.919425Z\",\"iopub.execute_input\":\"2022-10-22T12:27:19.919755Z\",\"iopub.status.idle\":\"2022-10-22T12:27:20.518612Z\",\"shell.execute_reply.started\":\"2022-10-22T12:27:19.919725Z\",\"shell.execute_reply\":\"2022-10-22T12:27:20.517411Z\"}}\nsorted_labels = sorted(\n    labels,\n    key=lambda name: (name[1:], name[0])\n)\nprint(metrics.flat_classification_report(\n    y_valid, y_pred, labels=sorted_labels, digits=3\n))\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2022-10-22T12:28:47.516017Z\",\"iopub.execute_input\":\"2022-10-22T12:28:47.516461Z\",\"iopub.status.idle\":\"2022-10-22T12:28:47.523045Z\",\"shell.execute_reply.started\":\"2022-10-22T12:28:47.516423Z\",\"shell.execute_reply\":\"2022-10-22T12:28:47.522122Z\"}}\nfor a, b in zip(valid_sentences[0], crf.predict([X_valid[0]])[0]):\n    print(a, b)\n\n# %% [code]\n","metadata":{"_uuid":"f0c045c4-90f7-4938-95fa-0d309694cfeb","_cell_guid":"35f71579-484e-43e5-acf0-65cc2dd2447e","collapsed":false,"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-10-22T12:34:19.407700Z","iopub.execute_input":"2022-10-22T12:34:19.408115Z","iopub.status.idle":"2022-10-22T12:35:32.896429Z","shell.execute_reply.started":"2022-10-22T12:34:19.408055Z","shell.execute_reply":"2022-10-22T12:35:32.895050Z"},"trusted":true},"execution_count":36,"outputs":[{"name":"stdout","text":"  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current\n                                 Dload  Upload   Total   Spent    Left  Speed\n100 3206k  100 3206k    0     0  17.3M      0 --:--:-- --:--:-- --:--:-- 17.2M\n  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current\n                                 Dload  Upload   Total   Spent    Left  Speed\n100  808k  100  808k    0     0  5146k      0 --:--:-- --:--:-- --:--:-- 5146k\n  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current\n                                 Dload  Upload   Total   Spent    Left  Speed\n100  730k  100  730k    0     0  4623k      0 --:--:-- --:--:-- --:--:-- 4623k\n-DOCSTART- -X- -X- O\n\nEU NNP B-NP B-ORG\nrejects VBZ B-VP O\nGerman JJ B-NP B-MISC\nLength of training set: 14041\nLength of validation set: 3250\n9\n9\n{'bias': 1.0, 'word.lower()': 'german', 'word[-3:]': 'man', 'word[-2:]': 'an', 'word.isupper()': False, 'word.istitle()': True, 'word.isdigit()': False, 'postag': 'JJ', 'postag[:2]': 'JJ', '-1:word.lower()': 'rejects', '-1:word.istitle()': False, '-1:word.isupper()': False, '-1:postag': 'VBZ', '-1:postag[:2]': 'VB', '+1:word.lower()': 'call', '+1:word.istitle()': False, '+1:word.isupper()': False, '+1:postag': 'NN', '+1:postag[:2]': 'NN'}\nB-MISC\nCollecting sklearn_crfsuite\n  Cloning https://github.com/MeMartijn/updated-sklearn-crfsuite.git to /tmp/pip-install-nfo6gvwl/sklearn-crfsuite_e9331dd3bb0c4b618bbcc409a409d34b\n  Running command git clone --filter=blob:none --quiet https://github.com/MeMartijn/updated-sklearn-crfsuite.git /tmp/pip-install-nfo6gvwl/sklearn-crfsuite_e9331dd3bb0c4b618bbcc409a409d34b\n  Resolved https://github.com/MeMartijn/updated-sklearn-crfsuite.git to commit 675038761b4405f04691a83339d04903790e2b95\n  Preparing metadata (setup.py) ... \u001b[?25ldone\n\u001b[?25hRequirement already satisfied: tqdm>=2.0 in /opt/conda/lib/python3.7/site-packages (from sklearn_crfsuite) (4.64.0)\nRequirement already satisfied: six in /opt/conda/lib/python3.7/site-packages (from sklearn_crfsuite) (1.15.0)\nRequirement already satisfied: tabulate in /opt/conda/lib/python3.7/site-packages (from sklearn_crfsuite) (0.9.0)\nRequirement already satisfied: python-crfsuite>=0.8.3 in /opt/conda/lib/python3.7/site-packages (from sklearn_crfsuite) (0.9.8)\n\u001b[33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv\u001b[0m\u001b[33m\n\u001b[0m","output_type":"stream"},{"name":"stderr","text":"loading training data to CRFsuite: 100%|██████████| 14041/14041 [00:03<00:00, 4021.61it/s]\n","output_type":"stream"},{"name":"stdout","text":"\nFeature generation\ntype: CRF1d\nfeature.minfreq: 0.000000\nfeature.possible_states: 0\nfeature.possible_transitions: 1\n0....1....2....3....4....5....6....7....8....9....10\nNumber of features: 86687\nSeconds required: 0.613\n\nL-BFGS optimization\nc1: 0.100000\nc2: 0.100000\nnum_memories: 6\nmax_iterations: 100\nepsilon: 0.000010\nstop: 10\ndelta: 0.000010\nlinesearch: MoreThuente\nlinesearch.max_iterations: 20\n\nIter 1   time=0.92  loss=232367.34 active=86384 feature_norm=1.00\nIter 2   time=0.45  loss=217022.74 active=83848 feature_norm=3.45\nIter 3   time=0.45  loss=161383.49 active=83844 feature_norm=2.99\nIter 4   time=1.33  loss=119552.24 active=83579 feature_norm=2.74\nIter 5   time=0.45  loss=94630.26 active=86129 feature_norm=3.20\nIter 6   time=0.45  loss=88330.83 active=85472 feature_norm=3.56\nIter 7   time=0.46  loss=71522.54 active=80105 feature_norm=5.12\nIter 8   time=0.45  loss=61055.57 active=64864 feature_norm=6.38\nIter 9   time=0.45  loss=54560.88 active=61779 feature_norm=7.80\nIter 10  time=0.45  loss=51008.46 active=60961 feature_norm=8.55\nIter 11  time=0.45  loss=47614.30 active=59392 feature_norm=9.55\nIter 12  time=0.45  loss=44079.27 active=57533 feature_norm=11.27\nIter 13  time=0.45  loss=40537.62 active=56452 feature_norm=13.05\nIter 14  time=0.44  loss=37682.10 active=54690 feature_norm=15.26\nIter 15  time=0.45  loss=35267.10 active=53295 feature_norm=17.32\nIter 16  time=0.45  loss=32895.02 active=52239 feature_norm=19.83\nIter 17  time=0.45  loss=30522.84 active=51527 feature_norm=23.17\nIter 18  time=0.46  loss=27634.95 active=50530 feature_norm=27.28\nIter 19  time=0.45  loss=24968.88 active=48502 feature_norm=33.06\nIter 20  time=0.45  loss=22881.99 active=47935 feature_norm=37.60\nIter 21  time=0.44  loss=20896.11 active=47474 feature_norm=42.08\nIter 22  time=0.46  loss=18504.24 active=45677 feature_norm=51.18\nIter 23  time=0.89  loss=17597.41 active=45388 feature_norm=57.24\nIter 24  time=0.45  loss=16002.47 active=45240 feature_norm=62.08\nIter 25  time=0.45  loss=14868.76 active=44778 feature_norm=67.72\nIter 26  time=0.45  loss=13354.86 active=43316 feature_norm=78.18\nIter 27  time=0.45  loss=12070.60 active=42741 feature_norm=87.33\nIter 28  time=0.45  loss=11258.35 active=42789 feature_norm=91.79\nIter 29  time=0.45  loss=10316.97 active=41681 feature_norm=99.70\nIter 30  time=0.45  loss=9654.87  active=40556 feature_norm=107.03\nIter 31  time=0.46  loss=8989.36  active=40400 feature_norm=113.18\nIter 32  time=0.44  loss=8587.66  active=40020 feature_norm=117.52\nIter 33  time=0.45  loss=8090.88  active=38227 feature_norm=124.06\nIter 34  time=0.45  loss=7725.09  active=36713 feature_norm=127.58\nIter 35  time=0.45  loss=7488.86  active=36247 feature_norm=130.44\nIter 36  time=0.45  loss=7256.28  active=35064 feature_norm=133.41\nIter 37  time=0.45  loss=7071.11  active=34702 feature_norm=135.91\nIter 38  time=0.44  loss=6941.14  active=34516 feature_norm=137.56\nIter 39  time=0.45  loss=6842.52  active=34206 feature_norm=139.03\nIter 40  time=0.45  loss=6760.20  active=32448 feature_norm=141.59\nIter 41  time=0.44  loss=6698.97  active=32415 feature_norm=142.19\nIter 42  time=0.45  loss=6658.05  active=31926 feature_norm=142.86\nIter 43  time=0.45  loss=6610.85  active=31126 feature_norm=144.18\nIter 44  time=0.45  loss=6588.73  active=30973 feature_norm=144.67\nIter 45  time=0.45  loss=6555.18  active=31095 feature_norm=144.64\nIter 46  time=0.47  loss=6539.32  active=30917 feature_norm=144.69\nIter 47  time=0.44  loss=6516.53  active=30564 feature_norm=144.78\nIter 48  time=0.46  loss=6480.51  active=29992 feature_norm=144.60\nIter 49  time=0.89  loss=6469.61  active=29661 feature_norm=144.81\nIter 50  time=0.45  loss=6444.13  active=29557 feature_norm=144.69\nIter 51  time=0.45  loss=6426.50  active=29168 feature_norm=144.78\nIter 52  time=0.45  loss=6412.01  active=28733 feature_norm=144.99\nIter 53  time=0.44  loss=6393.27  active=28679 feature_norm=145.16\nIter 54  time=0.48  loss=6384.95  active=28550 feature_norm=145.20\nIter 55  time=0.46  loss=6373.18  active=28316 feature_norm=145.26\nIter 56  time=0.46  loss=6361.27  active=28130 feature_norm=145.28\nIter 57  time=0.45  loss=6351.65  active=28051 feature_norm=145.30\nIter 58  time=0.45  loss=6343.71  active=27938 feature_norm=145.29\nIter 59  time=0.45  loss=6334.07  active=27787 feature_norm=145.26\nIter 60  time=0.45  loss=6326.09  active=27614 feature_norm=145.20\nIter 61  time=0.46  loss=6319.18  active=27478 feature_norm=145.20\nIter 62  time=0.45  loss=6312.73  active=27325 feature_norm=145.13\nIter 63  time=0.45  loss=6306.90  active=27180 feature_norm=145.10\nIter 64  time=0.45  loss=6301.91  active=27093 feature_norm=145.06\nIter 65  time=0.45  loss=6296.34  active=26976 feature_norm=145.02\nIter 66  time=0.45  loss=6290.52  active=26843 feature_norm=144.94\nIter 67  time=0.46  loss=6285.49  active=26729 feature_norm=144.91\nIter 68  time=0.45  loss=6281.28  active=26658 feature_norm=144.87\nIter 69  time=0.46  loss=6276.98  active=26588 feature_norm=144.85\nIter 70  time=0.45  loss=6272.96  active=26513 feature_norm=144.79\nIter 71  time=0.45  loss=6269.32  active=26423 feature_norm=144.77\nIter 72  time=0.45  loss=6265.30  active=26397 feature_norm=144.73\nIter 73  time=0.45  loss=6262.06  active=26355 feature_norm=144.73\nIter 74  time=0.45  loss=6258.82  active=26331 feature_norm=144.70\nIter 75  time=0.44  loss=6256.74  active=26259 feature_norm=144.69\nIter 76  time=0.45  loss=6253.72  active=26221 feature_norm=144.66\nIter 77  time=0.45  loss=6251.50  active=26208 feature_norm=144.66\nIter 78  time=0.44  loss=6249.27  active=26187 feature_norm=144.65\nIter 79  time=0.45  loss=6247.05  active=26173 feature_norm=144.65\nIter 80  time=0.45  loss=6245.59  active=26132 feature_norm=144.64\nIter 81  time=0.45  loss=6243.33  active=26091 feature_norm=144.65\nIter 82  time=0.44  loss=6241.47  active=26070 feature_norm=144.64\nIter 83  time=0.44  loss=6239.33  active=26035 feature_norm=144.65\nIter 84  time=0.44  loss=6238.09  active=26022 feature_norm=144.65\nIter 85  time=0.45  loss=6236.20  active=25989 feature_norm=144.65\nIter 86  time=0.45  loss=6235.04  active=25974 feature_norm=144.65\nIter 87  time=0.45  loss=6233.33  active=25950 feature_norm=144.66\nIter 88  time=0.45  loss=6231.97  active=25940 feature_norm=144.66\nIter 89  time=0.45  loss=6230.31  active=25914 feature_norm=144.67\nIter 90  time=0.44  loss=6229.00  active=25911 feature_norm=144.67\nIter 91  time=0.45  loss=6227.80  active=25886 feature_norm=144.68\nIter 92  time=0.45  loss=6226.55  active=25879 feature_norm=144.68\nIter 93  time=0.45  loss=6225.35  active=25862 feature_norm=144.70\nIter 94  time=0.46  loss=6224.33  active=25846 feature_norm=144.69\nIter 95  time=0.45  loss=6223.16  active=25824 feature_norm=144.71\nIter 96  time=0.45  loss=6222.17  active=25820 feature_norm=144.71\nIter 97  time=0.45  loss=6221.06  active=25799 feature_norm=144.72\nIter 98  time=0.45  loss=6220.18  active=25783 feature_norm=144.72\nIter 99  time=0.45  loss=6219.09  active=25750 feature_norm=144.74\nIter 100 time=0.45  loss=6218.16  active=25761 feature_norm=144.74\nL-BFGS terminated with the maximum number of iterations\nTotal seconds required for training: 47.190\n\nStoring the model\nNumber of active features: 25761 (86687)\nNumber of active attributes: 16414 (68166)\nNumber of active labels: 9 (9)\nWriting labels\nWriting attributes\nWriting feature references for transitions\nWriting feature references for attributes\nSeconds required: 0.018\n\n['B-ORG', 'O', 'B-MISC', 'B-PER', 'I-PER', 'B-LOC', 'I-ORG', 'I-MISC', 'I-LOC']\n0.8783682328710216\n0.9004358445802979\n0.8590732198364325\n              precision    recall  f1-score   support\n\n           O      0.991     0.997     0.994     42759\n       B-LOC      0.914     0.876     0.894      1837\n       I-LOC      0.878     0.786     0.830       257\n      B-MISC      0.926     0.839     0.881       922\n      I-MISC      0.885     0.737     0.804       346\n       B-ORG      0.851     0.809     0.830      1341\n       I-ORG      0.817     0.824     0.820       751\n       B-PER      0.901     0.908     0.905      1842\n       I-PER      0.941     0.955     0.948      1307\n\n    accuracy                          0.975     51362\n   macro avg      0.900     0.859     0.878     51362\nweighted avg      0.975     0.975     0.975     51362\n\n('CRICKET', 'NNP', 'O') O\n('-', ':', 'O') O\n('LEICESTERSHIRE', 'NNP', 'B-ORG') B-ORG\n('TAKE', 'NNP', 'O') O\n('OVER', 'IN', 'O') O\n('AT', 'NNP', 'O') O\n('TOP', 'NNP', 'O') O\n('AFTER', 'NNP', 'O') O\n('INNINGS', 'NNP', 'O') O\n('VICTORY', 'NN', 'O') O\n('.', '.', 'O') O\n","output_type":"stream"}]}]}