from numpy import *
import sys
import matplotlib.pyplot as plt
patient_visit_dt = dtype([('study','U10'),('country','U10'),('txgroup','U10'),('patientid', float64),('visitday', int32),('xvalues',float64,(30)),('panss',float64)])

def data_loader(fname):
    #Study	Country	PatientID	SiteID	RaterID	AssessmentID	TxGroup	VisitDay	P1	P2	P3	P4	P5	P6	P7	N1	N2	N3	N4	N5	N6	N7	G1	G2	G3	G4	G5	G6	
    # 0     1       2           3       4       5               6       7           8   9   10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27
    # G7	G8	G9	G10	G11	G12	G13	G14	G15	G16	PANSS_Total
    # 28    29  30  31  32  33  34  35  36  37  38

    #data_a = loadtxt(fname,skiprows=1, usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38), delimiter=',')
    cata_data_a = loadtxt(fname,skiprows=1, usecols=(0,1,6), delimiter=',', dtype=str )
    quant_data_a = loadtxt(fname,skiprows=1, usecols=(2,3,4,5,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38), delimiter=',', dtype=float64)
    #print(cata_data_a, quant_data_a)
    #print(cata_data_a.shape, quant_data_a.shape)
    #print(type(cata_data_a), type(quant_data_a))

    
    data_a = empty(quant_data_a.shape[0], dtype=patient_visit_dt)
    #print(data_a.shape)

    #raise SystemExit
    #visit_a = array((visit_a[['study', 'country', 'txgroup']],visit_a['xvalues'],visit_a['yvalues']), dtype = patient_visit_dt)
    #x_v = quant_data_a[:,5:35]
    
    #y_v = quant_data_a[:,35]
    #print(data_a[['study', 'country', 'txgroup']].shape, cata_data_a.shape)
    data_a['study']  = cata_data_a[:, 0]
    data_a['country']  = cata_data_a[:, 1]
    data_a['txgroup']  = cata_data_a[:, 2]
    data_a['patientid'] = quant_data_a[:,0]
    data_a['visitday'] = quant_data_a[:,4]
    #print("eeeee", quant_data_a[:,4])
    #print("aaaa", quant_data_a[:,0])

    #print(quant_data_a[:,5:35])
    #print(quant_data_a[:,35])

    data_a['xvalues'] = quant_data_a[:,5:35]
    data_a['panss'] = quant_data_a[:,35]
    #print(data_a['xvalues'].shape, data_a['panss'].shape)
    #print(data_a)

    #data_a[['study', 'country', 'txgroup']] = tuple(cata_data_a[:, i] for i in range(3))
    #print('!!', cata_data_a[:2]) 
    #data_a[['study', ] = cata_data_a[0, :2]
    #cata_data_a[:, :2] 

    #visit_a['xvalues'] = x_v
    #visit_a['yvalues'] = y_v
    #visit_a[['country', 'study']] = cata_data_a[:,0], cata_data_a[:,1]
    return data_a, cata_data_a, quant_data_a

def trial():
    num_l = []
    for i in range(100):
        num_l.append(i)
        #print(i)
    return num_l

def find_patients(data_a):
    d = {}
    #rows = range(data_a.shape[0])
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
    print("test", patient_d)
    return patient_d

def seperate_control_treatment(patient_d):
    control_l = []
    treatment_l = []

    for patient_a in patient_d.values():
        if patient_a[0]['txgroup'] == '"Control"':
            control_l.append(patient_a)
        else:
            treatment_l.append(patient_a)
    #print(control_l, treatment_l)
    return control_l, treatment_l

def plot_patient_data(data_l):
    print(type(data_l))
    print(len(data_l))

    fig, ax = plt.subplots()

    for patient_a in data_l:
        
        x_l = patient_a['visitday']
        y_l = patient_a['panss']
        ax.plot(x_l, y_l)
    plt.xlabel('Visit Day')
    plt.ylabel('PANSS Total')
    plt.title('Every Treatment Group Patient and Their PANSS Score Over Time')
    plt.show()
def filter_patients(patient_d, day_limit=126):
    print(type(patient_d))
    print(patient_d)
    #delete_patient_l = [patientid for patientid, patient_a in patient_d.items() if patient_a[-1]['visitday'] < day_limit]
    delete_patient_l = [patientid for patientid, patient_a in patient_d.items() if patient_a[-1]['visitday'] - patient_a[0]['visitday'] < day_limit]
    for patientid in delete_patient_l:
        del patient_d[patientid]
def fit_linearmodel():
    pass
def main():
    #data_a = hstack([data_loader(fname) for fname in sys.argv[1:]])

    data_a,cata_data_a, quant_data_a = data_loader("Study_E.csv")
    patientid_v = data_a['patientid']
    upatientid_v = unique(patientid_v)
    print('Patient id:', upatientid_v.shape[0])

    print("just viewing", patientid_v)
    #print(da)

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

    control_l, treatment_l = seperate_control_treatment(patient_d)
    print("control", control_l)
    print('------------')
    print('------------')
    print('------------')
    print('------------')
    print('------------')
    print("treatment", treatment_l)
    
    plot_patient_data(treatment_l)
    plot_patient_data(control_l)
    raise SystemExit
    plt.xlabel('Visit Day')
    plt.ylabel('PANSS Total')
    plt.title('Every Treatment Group Patient and Their PANSS Score Over Time')
if __name__ == '__main__':
    main()