import loader_2015_epoch as loader2015


training_subject_range = [
    (1, 2),
    (1, 4),
    (1, 11),
    (1, 31)
]


def produce_rf():
    import train_rf
    print("produce_rf: start")
    for i in range(len(training_subject_range)):
        subject_count = training_subject_range[i][1] - training_subject_range[i][0]
        modle_output_path = f"./models/rf{subject_count}.pkl"
        report_output_path = f"./report/rf{subject_count}.txt"
        X_train, y_train = loader2015.load(training_subject_range[i][0], training_subject_range[i][1])
        train_rf.train(modle_output_path, report_output_path, X_train, y_train)
        print(f"produce_rf: finish subject_num = {subject_count}")


def produce_lda():
    import train_lda
    print("produce_lda: start")
    for i in range(len(training_subject_range)):
        subject_count = training_subject_range[i][1] - training_subject_range[i][0]
        modle_output_path = f"./models/lda{subject_count}.pkl"
        report_output_path = f"./report/lda{subject_count}.txt"
        X_train, y_train = loader2015.load(training_subject_range[i][0], training_subject_range[i][1])
        train_lda.train(modle_output_path, report_output_path, X_train, y_train)
        print(f"produce_lda: finish subject_num = {subject_count}")

def produce_cnn():
    import train_cnn
    print("produce_cnn: start")
    for i in range(len(training_subject_range)):
        subject_count = training_subject_range[i][1] - training_subject_range[i][0]
        modle_output_path = f"./models/cnn{subject_count}.pth"
        report_output_path = f"./report/cnn{subject_count}.txt"
        X_train, y_train = loader2015.load(training_subject_range[i][0], training_subject_range[i][1])
        train_cnn.train(modle_output_path, report_output_path, X_train, y_train)
        print(f"produce_cnn: finish subject_num = {subject_count}")
        

#produce_rf()
#produce_lda()
produce_cnn()