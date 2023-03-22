import loader_2015_epoch as loader2015
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import os
import numpy as np
import pickle

pretrain_subject_range = [
    (0, 0),
    (1, 2),
    (1, 4),
    (1, 11),
    (1, 31)
]
test_pretrain_sample = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]
test_start_epoch_idx = test_pretrain_sample[-1]
test_subject_range = [31, 44]
result = dict()
result_obj_path = "./report/result2015.pkl"
result_txt_path = "./report/result2015.txt"

def save_report(y_gt, y_pred, clf_name, pretrain_subject_count, test_pretrain_sample_count):
    global result
    report = dict()
    report_id = f"{clf_name}_{pretrain_subject_count}_{test_pretrain_sample_count}"
    report["f1"] = f1_score(y_gt, y_pred, average="macro")
    report['acc'] = sum(y_gt == y_pred) / y_gt.shape[0]
    report['clf_name'] = clf_name
    report['pretrain_subject_count'] = pretrain_subject_count
    report['test_pretrain_sample_count'] = test_pretrain_sample_count
    report['report'] = ""
    report_txt = f"{report_id}\n"
    report_txt += "tn, fp, fn, tp: " + "  ".join([str(i) for i in confusion_matrix(y_gt, y_pred).ravel().tolist()]) + "\n"
    report_txt += "f1 score: " + f1_score(y_gt, y_pred).astype(str) + "\n"
    report_txt += "accuracy: " + (report['acc']).astype(str) + "\n"
    report_txt += classification_report(y_gt, y_pred)
    report['report'] = report_txt
    result[report_id] = report
    with open(result_obj_path, "wb") as f:
        pickle.dump(result, f)
    _save_report_txt()
    return report['f1'], report['acc']
        
def _save_report_txt():
    _test_init()
    all_report_txt = ""
    for report_id in result:
        report = result[report_id]
        report_txt = report['report']
        all_report_txt += report_txt + "\n\n"
    with open(result_txt_path, "w") as f:
        f.write(all_report_txt)

def _test_init():
    global result
    if (len(result) == 0) and os.path.exists(result_obj_path):
        with open(result_obj_path, "rb") as f:
            result = pickle.load(f)
        
def test_rf():
    import test_rf as tester
    _test_init()
    clf_name = "rf"
    print(f"test_{clf_name}: start")
    for p_range in pretrain_subject_range:
        model_path = None
        pretrain_cnt = p_range[1] - p_range[0]
        X_train_pre = None
        y_train_pre = np.array([])
        if pretrain_cnt != 0:
            X_train_pre, y_train_pre = loader2015.load(p_range[0], p_range[1])
        for test_cnt in test_pretrain_sample:
            all_y_pred = np.array([])
            all_y_gt = np.array([])
            if test_cnt == 0 and pretrain_cnt == 0:
                continue
            
            for sub2015_num in range(test_subject_range[0], test_subject_range[1]):
                X_train, y_train = loader2015.load(sub2015_num, sub2015_num + 1)
                X_test = X_train[test_start_epoch_idx:]
                y_test = y_train[test_start_epoch_idx:]
                
                X_train = X_train[:test_start_epoch_idx] if X_train_pre is None else np.concatenate((X_train_pre, X_train[:test_start_epoch_idx]))
                y_train = np.concatenate((y_train_pre, y_train[:test_start_epoch_idx]))
                
                y_pred = tester.train_and_test(X_train, y_train, X_test)
                all_y_pred = np.concatenate((all_y_pred, y_pred))
                all_y_gt = np.concatenate((all_y_gt, y_test))
            f1, acc = save_report(all_y_gt, all_y_pred, clf_name, pretrain_cnt, test_cnt)
            print(f"test_{clf_name}: finish pretrain_cnt = {pretrain_cnt}, test_cnt = {test_cnt}")
            print(f"f1 score: {f1}, accuracy: {acc}")


def test_svm():
    import test_svm as tester
    _test_init()
    clf_name = "svm"
    print(f"test_{clf_name}: start")
    for p_range in pretrain_subject_range:
        model_path = None
        pretrain_cnt = p_range[1] - p_range[0]
        X_train_pre = None
        y_train_pre = np.array([])
        if pretrain_cnt != 0:
            X_train_pre, y_train_pre = loader2015.load(p_range[0], p_range[1])
        for test_cnt in test_pretrain_sample:
            all_y_pred = np.array([])
            all_y_gt = np.array([])
            if test_cnt == 0 and pretrain_cnt == 0:
                continue
            
            for sub2015_num in range(test_subject_range[0], test_subject_range[1]):
                X_train, y_train = loader2015.load(sub2015_num, sub2015_num + 1)
                X_test = X_train[test_start_epoch_idx:]
                y_test = y_train[test_start_epoch_idx:]
                
                X_train = X_train[:test_cnt]
                y_train = y_train[:test_cnt]
                
                y_pred = tester.train_and_test(X_train_pre, y_train_pre, X_train, y_train, X_test)
                all_y_pred = np.concatenate((all_y_pred, y_pred))
                all_y_gt = np.concatenate((all_y_gt, y_test))
            f1, acc = save_report(all_y_gt, all_y_pred, clf_name, pretrain_cnt, test_cnt)
            print(f"test_{clf_name}: finish pretrain_cnt = {pretrain_cnt}, test_cnt = {test_cnt}")
            print(f"f1 score: {f1}, accuracy: {acc}")


def test_lda():
    import test_lda as tester
    _test_init()
    clf_name = "lda"
    print("test_lda: start")
    for p_range in pretrain_subject_range:
        model_path = None
        pretrain_cnt = p_range[1] - p_range[0]
        X_train_pre = None
        y_train_pre = np.array([])
        if pretrain_cnt != 0:
            X_train_pre, y_train_pre = loader2015.load(p_range[0], p_range[1])
        for test_cnt in test_pretrain_sample:
            all_y_pred = np.array([])
            all_y_gt = np.array([])
            if test_cnt == 0 and pretrain_cnt == 0:
                continue
            
            for sub2015_num in range(test_subject_range[0], test_subject_range[1]):
                X_train, y_train = loader2015.load(sub2015_num, sub2015_num + 1)
                X_test = X_train[test_start_epoch_idx:]
                y_test = y_train[test_start_epoch_idx:]
                
                X_train = X_train[:test_cnt]
                y_train = y_train[:test_cnt]
                
                y_pred = tester.train_and_test(X_train_pre, y_train_pre, X_train, y_train, X_test)
                all_y_pred = np.concatenate((all_y_pred, y_pred))
                all_y_gt = np.concatenate((all_y_gt, y_test))
            f1, acc = save_report(all_y_gt, all_y_pred, clf_name, pretrain_cnt, test_cnt)
            print(f"test_{clf_name}: finish pretrain_cnt = {pretrain_cnt}, test_cnt = {test_cnt}")
            print(f"f1 score: {f1}, accuracy: {acc}")

def test_cnn():
    import test_cnn as tester
    _test_init()
    clf_name = "cnn"
    print(f"test_{clf_name}: start")
    for p_range in pretrain_subject_range:
        pretrain_cnt = p_range[1] - p_range[0]
        model_path = f"./models/cnn{pretrain_cnt}.pth"
        X_train_pre = None
        y_train_pre = np.array([])
        if pretrain_cnt != 0:
            X_train_pre, y_train_pre = loader2015.load(p_range[0], p_range[1])
        else:
            model_path = None
        for test_cnt in test_pretrain_sample:
            all_y_pred = np.array([])
            all_y_gt = np.array([])
            if test_cnt == 0 and pretrain_cnt == 0:
                continue
            
            for sub2015_num in range(test_subject_range[0], test_subject_range[1]):
                X_train = None
                y_train = None
                X_train_test, y_train_test = loader2015.load(sub2015_num, sub2015_num + 1)
                X_test = X_train_test[test_start_epoch_idx:]
                y_test = y_train_test[test_start_epoch_idx:]
                if test_cnt == 0:
                    X_train = X_train_pre
                    y_train = y_train_pre
                else:
                    X_train = X_train_test[:test_cnt] if pretrain_cnt == 0 else np.concatenate((X_train_pre, X_train_test[:test_cnt]))
                    y_train = np.concatenate((y_train_pre, y_train_test[:test_cnt]))
                
                y_pred = tester.train_and_test(model_path, X_train, y_train, X_test)
                all_y_pred = np.concatenate((all_y_pred, y_pred))
                all_y_gt = np.concatenate((all_y_gt, y_test))
            f1, acc = save_report(all_y_gt, all_y_pred, clf_name, pretrain_cnt, test_cnt)
            print(f"test_{clf_name}: finish pretrain_cnt = {pretrain_cnt}, test_cnt = {test_cnt}")
            print(f"f1 score: {f1}, accuracy: {acc}")







test_svm()
test_rf()