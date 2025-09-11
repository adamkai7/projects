from cgi import test
from numpy import *
import sys
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
patient_visit_dt = dtype([('study','U10'),('country','U10'),('txgroup','U10'),('patientid', float64),('visitday', int32),('xvalues',float64,(7)),('panss',float64), ('leadstatus','U10')])

def data_loader(fname):
    #Study	Country	PatientID	SiteID	RaterID	AssessmentID	TxGroup	VisitDay	P1	P2	P3	P4	P5	P6	P7	N1	N2	N3	N4	N5	N6	N7	G1	G2	G3	G4	G5	G6	
    # 0     1       2           3       4       5               6       7           8   9   10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27
    # G7	G8	G9	G10	G11	G12	G13	G14	G15	G16	PANSS_Total
    # 28    29  30  31  32  33  34  35  36  37  38

    cata_data_a = loadtxt(fname,skiprows=1, usecols=(0,1,6,39), delimiter=',', dtype=str )
    #quant_data_a = loadtxt(fname,skiprows=1, usecols=(2,3,4,5,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38), delimiter=',', dtype=float64)
    quant_data_a = loadtxt(fname,skiprows=1, usecols=(15,16,17,18,19,20,21,35), delimiter=',', dtype=float64)


    
    data_a = empty(quant_data_a.shape[0], dtype=patient_visit_dt)
    #print(data_a.shape)

   
    data_a['study']  = cata_data_a[:, 0]
    data_a['country']  = cata_data_a[:, 1]
    data_a['txgroup']  = cata_data_a[:, 2]
    data_a['patientid'] = quant_data_a[:,0]
    data_a['visitday'] = quant_data_a[:,4]


    data_a['xvalues'] = quant_data_a[:,0:7]
    data_a['panss'] = quant_data_a[:,7]
    data_a['leadstatus'] = cata_data_a[:,3]
    return data_a

def trial():
    num_l = []
    for i in range(100):
        num_l.append(i)
    return num_l

def find_patients(data_a):
    d = {}
    for i in range(data_a.shape[0]):
        key = data_a[i]['patientid']  
        if key in d:
            d[key].append(i)
        else:
            d[key] = [i]

    patient_d = {}
    for patientid, index_l in d.items():
        patient_a = data_a[index_l]
        index_v = patient_a['visitday'].argsort()       
        patient_d[patientid] = patient_a[index_v]

    print(type(patient_d))
    return patient_d

def seperate_control_treatment(patient_d):
    control_d = {}
    treatment_d = {}

    for patient_a in patient_d.values():

        if patient_a[0]['txgroup'] == '"Control"':
            control_d[patient_a[0]['patientid']] = patient_a
        else:
            #print('test')
            treatment_d[patient_a[0]['patientid']] = patient_a
    
    return control_d, treatment_d
        

def plot_patient_data(data_l):
    print(type(data_l))
    print(len(data_l))

    fig, ax = plt.subplots()

    for patient_a in data_l:
        
        x_l = patient_a['visitday']
        y_l = patient_a['panss']
        ax.plot(x_l, y_l)
    plt.show()

def filter_patients(patient_d, day_limit=126):
    print(type(patient_d))
    print(patient_d)
    #delete_patient_l = [patientid for patientid, patient_a in patient_d.items() if patient_a[-1]['visitday'] < day_limit]
    delete_patient_l = [patientid for patientid, patient_a in patient_d.items() if patient_a[-1]['visitday'] - patient_a[0]['visitday'] < day_limit]
    for patientid in delete_patient_l:
        del patient_d[patientid]

def difference_in_fields(patient_d, field_name):
    difference_values_l = []
    for patient_a in patient_d.values():
        dif1 = patient_a[-1][field_name]
        dif2 = patient_a[0][field_name]
        dif = dif1-dif2
        #dif = patient_a[-1]['panss'] - patient_a[1]['panss']
        difference_values_l.append(dif)
    difference_values_a = array(difference_values_l)
    return difference_values_a

def difference_in_scores(patient_d):
    assert 0
    difference_values_l = []
    for patient_a in patient_d.values():
        dif1 = patient_a[-1]['panss']
        dif2 = patient_a[0]['panss']
        dif = dif1-dif2
        difference_values_l.append(dif)
    return difference_values_l

def difference_in_scores_stats(difference_values_l):
    mean_difference = mean(difference_values_l)
    standev_difference = std(difference_values_l)
    print(f'The mean difference is {mean_difference} and has a standard deviation of {standev_difference}')
    return mean_difference, standev_difference

def knn_pred(ztrain_a, train_data_a, ztest_a,test_data_a):
    xtrain_v = ztrain_a
    ytrain_v = train_data_a['leadstatus']
    xtest_v = ztest_a
    ytest_v = test_data_a['leadstatus']
    print('shapes', xtrain_v.shape, ytrain_v.shape, xtest_v.shape, ytest_v.shape)

    knn_model = KNeighborsClassifier(n_neighbors = 1)
    print('shapes', xtrain_v.shape, ytrain_v.shape, xtest_v.shape, ytest_v.shape)
    knn_model.fit(xtrain_v, ytrain_v)
    ytrainpred = knn_model.predict(xtrain_v)
    ypred_v = knn_model.predict(xtest_v)
    ypred_proba = knn_model.predict_proba(xtest_v)
    accuracy_train = ((ytrain_v==ytrainpred).sum())/ytrainpred.shape[0]
    accuracy_test = ((ytest_v==ypred_v).sum())/ytest_v.shape[0]

    print(f'Training predicted values{ytrainpred}')
    print(f'Test predicited values{ypred_v}')

    print(f'The training accuracy is: {accuracy_train} ')
    print(f'The test accyract is {accuracy_test}')
    return ypred_proba

def z_score_conver(data_a):
    train_data_a = data_a['xvalues']
    #print(train_data_a)
    mean_a = train_data_a.mean(axis=0)
    standev_a = train_data_a.std(axis=0)
    zscore = (train_data_a - mean_a)/standev_a
    #print(zscore)
    return zscore

def z_score_convert2(data_a, input_test_data_a):
    train_data_a = data_a['xvalues']
    test_data_a = input_test_data_a['xvalues']
    #print(train_data_a)
    train_mean_v = train_data_a.mean(axis=0)
    train_standev_v = train_data_a.std(axis=0)
    train_zscore_a = (train_data_a - train_mean_v)/train_standev_v
    test_zscore_a = (test_data_a - train_mean_v)/train_standev_v
    #print(zscore)
    return train_zscore_a, test_zscore_a

def time_shifter(patient_d):
    #adjusted_d = {}
    for patient_id, visitday in patient_d.items():
        time_zero = visitday[0]['visitday']
        visitday[0] = 0
        for other_days in visitday[1:]:
            other_days['visitday']-=time_zero

def desired_data(patient_d):
    predicted_data_l = []
    for patient_id, array_values_a in patient_d.items():
        last_array = array_values_a[-1]
        output_data = last_array['panss']
        predicted_data_l.append((patient_id, output_data))
    submitted_data_a = array(predicted_data_l, output_data)
    return submitted_data_a
    
def save_file(patient_data_a, fname):
    savetxt(fname, patient_data_a, delimiter=',', fmt='%d')

def main():
    data_a = hstack([data_loader(fname) for fname in sys.argv[1:]])
    train_data_a = hstack([data_loader(fname) for fname in sys.argv[1:-1]])
    test_data_a = data_loader(sys.argv[-1])
    print(f'Actual y train values{train_data_a}')
    print(f'Actual y test values{test_data_a}')
    # print("seeing train after load", train_data_a.shape)
    #print("seeing test after load", test_data_a.shape)

    patientid_v = data_a['patientid']
    upatientid_v = unique(patientid_v)
    #print('Patient id:', upatientid_v.shape[0])

    #print("just viewing", patientid_v)

    #print(da)
    patient_d = find_patients(data_a)

    #print('------------')

    control_d, treatment_d = seperate_control_treatment(patient_d)
    #print("view control people", control_d)
    #print(type(control_d))
    #print("view treatment people", treatment_d)
    #print(type(control_d))
    #print('------------')

    control_patientdiff = difference_in_fields(control_d, 'panss')
    control_patientdiffstats = difference_in_scores_stats(control_patientdiff)
    #print(type(control_patientdiff))

    treatment_patientdiff = difference_in_fields(treatment_d, 'panss')
    treatment_patientdiffstats = difference_in_scores_stats(treatment_patientdiff)

    #print('------------')
    

    print(f'Control Patients: {control_patientdiffstats}')
    print(f'Treatment Patients: {treatment_patientdiffstats}')

    print('------------')
    #print(patient_d)
    fname = "upload1.csv"
    upload_data_a = desired_data(patient_d)
    print(upload_data_a)
    save_file(upload_data_a, fname)

    print('------------')
    zscore_train_a, zscore_test_a = z_score_convert2(train_data_a, test_data_a)
    print("train z score shape:", zscore_train_a.shape)
    print("test z score shape:", zscore_test_a.shape)
    print("data_a shape:", data_a.shape)
    print("test_data_a shape:", test_data_a.shape)



    ypredproba_v = knn_pred(zscore_train_a, train_data_a, zscore_test_a, test_data_a)
    print(ypredproba_v)
    print('------------')

    raise SystemExit

    # day_v = data_a['visitday']
    # # uday_v = unique(day_v)
    # week_v = uday_v/7
    # print(uday_v)
    # print('------------')
    # print('------------')
    # print(week_v)
    # print('------------')

    # print(f'data_a:10 {data_a[10:]}')
    m_v = data_a['txgroup'] ==   '"Treatment'
    day_v = data_a[m_v]['visitday']
    print('treatment day', sorted(day_v))
    uday_v = unique(day_v)
    # week_v = uday_v//7
    # print(f'uday_v treatments: {uday_v}')
    # print(f'week_v treatments: {week_v}')
    # #week_v = int32(week_v)
    # print('week count:', list(zip(arange(week_v.shape[0]), bincount(week_v))))
    patient_d = find_patients(data_a)
    print('Before delete:', len(patient_d))
    filter_patients(patient_d, day_limit=105)
    print('After delete:', len(patient_d))

    #print(len(a))
    print('------------')
    #print(d)
    print(patient_d)

    control_d, treatment_d = seperate_control_treatment(patient_d)
    #print("control", control_l)
    print('------------')
    print('------------')
    print('------------')
    print('------------')
    print('------------')
    #print("treatment", treatment_l)
    #plot_patient_data(control_l)
    #plot_patient_data(treatment_l)
if __name__ == '__main__':
    main()