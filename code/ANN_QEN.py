
import sys
sys.path.append(r'~/code')
from func import *


# set random seeds for reproducibility
# torch.manual_seed(10)

model_train = True


data_used='train.csv'

# check if a GPU is available
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

save_loc=loc = '~/ANN_QEN/'


# Load training and test features and catches from CSV files.
feature_names = ['simyear','month','trygear','plotter','pc_sat','echocol','hullg',
                  'nav_acc_metres','lon','lat','imp1_hours',
                   'satig','local_tiger_effort','vcode','o_brdn','cday']
cat_cols = ['simyear','month','trygear','plotter','pc_sat','echocol','hullg',
                  'nav_acc_metres','vcode','o_brdn']
con_val=['lon', 'lat','imp1_hours','local_tiger_effort','satig','cday']
species="blueendeavour"

# load training data, test data. CPUE prediction data
tr = pd.read_csv(loc + data_used)
ts = pd.read_csv(loc + 'test.csv')
pred_data = pd.read_csv(loc + 'predDat.csv')
pred_data[species] = 0

# split independent and dependent variables
X_tr, y_tr = tr[feature_names], tr[species].values
X_ts, y_ts = ts[feature_names], ts[species].values
X_pred, y_pred = pred_data[feature_names], pred_data[species].values

train_row = X_tr.shape[0]
test_row = X_ts.shape[0]
pred_row = X_pred.shape[0]

#concat train, test, pred datasets into new_data
new_data = pd.concat([pd.DataFrame(X_tr), pd.DataFrame(X_ts), pd.DataFrame(X_pred)],axis=0)

new_data = new_data.reset_index(drop=True)

#normalize the continue varibales
new_data[con_val] = lon_lat_norm(X_tr[con_val], new_data[con_val])

# Conver tcategorical features to dummy features.

### onehotencoder categorical variables only
# Identify which columns are categorical

# Create a OneHotEncoder object
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
new_data_encoded = encoder.fit_transform(new_data[cat_cols])
column_names = encoder.get_feature_names_out(cat_cols)
new_data_encoded = pd.DataFrame(new_data_encoded, columns=column_names)

# Create a new DataFrame with the encoded categorical data and the continuous data
new_data = pd.concat([new_data_encoded, new_data.drop(cat_cols, axis=1)], axis=1)

# split into 3 parts
# N module
N_vars = ['lon', 'lat']
N_vars.extend(['simyear_' + str(val) for val in X_tr['simyear'].unique()])
N_vars.extend(['month_' + str(val) for val in X_tr['month'].unique()])
# ST.extend(['tiger_region_' + str(val) for val in ts['tiger_region'].unique()])

q_vars = ['satig']
q_vars.extend(['cday'])
q_vars.extend(['month_' + str(val) for val in X_tr['month'].unique()])
q_vars.extend(['trygear_' + str(val) for val in X_tr['trygear'].unique()])
q_vars.extend(['plotter_' + str(val) for val in X_tr['plotter'].unique()])
q_vars.extend(['pc_sat_' + str(val) for val in X_tr['pc_sat'].unique()])
q_vars.extend(['hullg_' + str(val) for val in X_tr['hullg'].unique()])
q_vars.extend(['nav_acc_metres_' + str(val) for val in X_tr['nav_acc_metres'].unique()])
q_vars.extend(['echocol_' + str(val) for val in X_tr['echocol'].unique()])
q_vars.extend(['vcode_' + str(val) for val in X_tr['vcode'].unique()])
q_vars.extend(['o_brdn_' + str(val) for val in X_tr['o_brdn'].unique()])

E_vars = ['imp1_hours']
E_vars.extend(['local_tiger_effort'])

N_mod, q_mod, e_mod = new_data[N_vars], new_data[q_vars], new_data[E_vars]

# split training and test dataset
# y_tr, y_val, y_ts = n2t(y_tr).unsqueeze(1), n2t(y_val).unsqueeze(1), n2t(y_ts).unsqueeze(1)
# X_tr_ST, X_val_ST, X_ts_ST, X_pred_ST = split_data_NPF(ST, train_row, val_row, test_row, pred_row)
# X_tr_q, X_val_q, X_ts_q, X_pred_q = split_data_NPF(q, train_row, val_row, test_row, pred_row)
# X_tr_effort, X_val_effort, X_ts_effort, X_pred_effort = split_data_NPF(effort, train_row, val_row, test_row,
#                                                                        pred_row)
criterion = nn.MSELoss()
y_tr, y_ts = n2t(y_tr).unsqueeze(1), n2t(y_ts).unsqueeze(1)
X_tr_N, X_ts_N, X_pred_N = split_data_NPF(N_mod, train_row, test_row, pred_row)
X_tr_q, X_ts_q, X_pred_q = split_data_NPF(q_mod, train_row, test_row, pred_row)
X_tr_e, X_ts_e, X_pred_e = split_data_NPF(e_mod, train_row, test_row,
                                                                       pred_row)
criterion = nn.MSELoss()

# torch.manual_seed(10)

# try 10 different random seeds
for seed in range(0,10):
    torch.manual_seed(seed)

    neuron=[64,64,64,1]
    print('neuron:', neuron)

    # model name
    NN_str = 'ANN_QEN_' +  '_' + str(seed)

    layer=len(neuron)

    # N module: three hidden layers, each with 64 neurons.
    mlp1 = nn.Sequential(
        nn.Linear(X_tr_N.shape[1], 64),
        nn.Sigmoid(),
        nn.Linear(64, 64),
        nn.BatchNorm1d(64),
        nn.Sigmoid(),
        nn.Linear(64, 64),
        nn.BatchNorm1d(64),
        nn.Sigmoid(),
        nn.Linear(64, 1),
        nn.Sigmoid()
    )

    # q module: Two hidden layers, each with 8 neurons.
    mlp2 = nn.Sequential(
        nn.Linear(X_tr_q.shape[1], 8),
        nn.Sigmoid(),
        nn.Linear(8, 8),
        nn.BatchNorm1d(8),
        nn.Sigmoid(),
        nn.Linear(8, 1),
        nn.Sigmoid()
    )
    # e module: Two hidden layers, each with 8 neurons.
    mlp3= nn.Sequential(
        nn.Linear(X_tr_e.shape[1], 8),
        nn.Sigmoid(),
        nn.Linear(8, 8),
        nn.BatchNorm1d(8),
        nn.Sigmoid(),
        nn.Linear(8, 1),
        nn.Sigmoid()
    )

    # combine 3 modules
    net = tw_EqB(mlp1, mlp2, mlp3)

    # move to GPU
    net = net.to(device)
    X_tr_N = X_tr_N.to(device)
    X_tr_q = X_tr_q.to(device)
    X_tr_e = X_tr_e.to(device)
    y_tr = y_tr.to(device)

    # test model
    print(net(X_tr_N, X_tr_q, X_tr_e)[0])

    dataset = DatasetWrapper_tw([X_tr_N, X_tr_q, X_tr_e], y_tr)

    # learning rate schedule
    LR_SCHEDULE = [
        (1, 0.01),
        (150, 0.001),
        (200, 0.0001),
        (300, 0.00005)
    ]
    optimizer = Adam(net.parameters(), lr=1)
    scheduler = lr_scheduler.LambdaLR(optimizer, lambda epoch: lr_schedule(epoch, LR_SCHEDULE))

    # decine loss function
    loss = MSELoss()
    # early stop
    early_stopping = EarlyStopping(patience=30, verbose=False,loc=save_loc,NN_str=NN_str)

    # training
    if  model_train:
        train_tw(net, optimizer, scheduler, loss, dataset, batch_size=1000, nepochs=10000,
                 val_data=[X_tr_N, X_tr_q, X_tr_e], val_y=y_tr,
                 test_data=[X_ts_N, X_ts_q, X_ts_e], test_y=y_ts,
                 device=device, scenario="NPF", NN_str=NN_str, early_stopping=early_stopping,
                 criterion=criterion)

    # load best model
    model_path = save_loc + 'model/' + str(NN_str) + '.pt'
    net.load_state_dict(torch.load(model_path))


    ######### prediction  #########
    with torch.no_grad():   # move back to CPUE to calculate R2 and mse.
        net=net.to(device)
        X_tr_N = X_tr_N.to(device)
        X_tr_q = X_tr_q.to(device)
        X_tr_e = X_tr_e.to(device)
        tr,others = net(X_tr_N, X_tr_q, X_tr_e)
        pred_tr =(tr).to(torch.device('cpu'))

        X_ts_N=X_ts_N.to(device)
        X_ts_q=X_ts_q.to(device)
        X_ts_e=X_ts_e.to(device)
        pred_ts,others = net(X_ts_N,X_ts_q,X_ts_e)
        pred_ts = pred_ts.to(torch.device('cpu'))

        X_pred_N=X_pred_N.to(device)
        X_pred_q=X_pred_q.to(device)
        X_pred_e=X_pred_e.to(device)
        pred_cpue,others = net(X_pred_N,X_pred_q,X_pred_e)
        pred_cpue = pred_cpue.to(torch.device('cpu'))

    y_tr=y_tr.to(torch.device('cpu'))
    y_ts=y_ts.to(torch.device('cpu'))

    R2 = r2_score(y_tr, pred_tr)
    mse_train = mean_squared_error(y_tr, pred_tr)
    R2_test = r2_score(y_ts, pred_ts)
    mse_test = mean_squared_error(y_ts, pred_ts)

    print( 'mse_train:', mse_train, 'R2_train:', R2, 'mse_test:', mse_test, 'R2_test:', R2_test)

    ac = [str(NN_str), R2, mse_train, R2_test, mse_test]

    # print out # of parameters
    total_params = sum(p.numel() for p in net.parameters())
    print(f"Total number of parameters: {total_params}")

