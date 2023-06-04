clc;clear;close all;	
%% load data
data_str = 'Well_A.xlsx' ;   %读取数据的路径 	
data = readmatrix(data_str);
gas_production = data(:,2);		
gas_production_cell = [];	
%% VMD Process
switch 'None'
    case 'VMD'
        if_process = 'VMD';
        t = 1:length(gas_production(:,end));
        deco_num = 4;	% decomposition dimension        
        figure
        [imf,res] = vmd(gas_production(:,end),'NumIMF',deco_num);	
        [p,q] = ndgrid(t,1:size(imf,2));	
        imf_L = [gas_production(:,end),imf,res];	
        P = [p,p(:,1:2)];	
        Q = [q(:,1),q+1,q(:,end)+2];
        plot3(P,Q,imf_L)	
        grid on	
        xlabel('Time Values')	
        ylabel('Mode Number')	
        zlabel('Mode Amplitude')
        
        for NN1=1:deco_num	
            gas_production_cell{1,NN1}=[gas_production(:,1:end-1),imf(:,NN1)];	
        end
    case 'None'
        if_process = '';
        gas_production_cell{1,1} = gas_production;
end
%% Training model
% Model parameters	
min_batchsize = 20;       % batchsize			
max_epoch = 30;           % epoch

list_cell =	{[1]};
x_mu_all  = [];
x_sig_all = [];
y_mu_all  = [];
y_sig_all = [];
for NUM_all = 1:length(gas_production_cell)	
    data_process = gas_production_cell{1,NUM_all};	
    [x_feature_label,y_feature_label] = timeseries_process2(data_process,1,10); 	
    [~,y_feature_label1] = timeseries_process2(gas_production,1,10);	
    index_label1 = 1:(size(x_feature_label,1)); index_label=index_label1;	
    spilt_ri = [6 2 2];	% divide training set, validation set and testing set
    train_num=round(spilt_ri(1)/(sum(spilt_ri))*size(x_feature_label,1));	
    vaild_num=round((spilt_ri(1)+spilt_ri(2))/(sum(spilt_ri))*size(x_feature_label,1));
    % 
    train_x_feature_label= x_feature_label(index_label(1:train_num),:);	
    train_y_feature_label= y_feature_label(index_label(1:train_num),:);	
    vaild_x_feature_label= x_feature_label(index_label(train_num+1:vaild_num),:);	
    vaild_y_feature_label= y_feature_label(index_label(train_num+1:vaild_num),:);	
    test_x_feature_label = x_feature_label(index_label(vaild_num+1:end),:);	
    test_y_feature_label = y_feature_label(index_label(vaild_num+1:end),:);	
	
    % training set	
    x_mu  = mean(train_x_feature_label);  
    x_sig = std(train_x_feature_label); 	
    train_x_feature_label_norm = (train_x_feature_label - x_mu) ./ x_sig;  
    y_mu  = mean(train_y_feature_label);  
    y_sig = std(train_y_feature_label); 	
    train_y_feature_label_norm = (train_y_feature_label - y_mu) ./ y_sig; 
    x_mu_all(NUM_all,:) = x_mu;
    x_sig_all(NUM_all,:)= x_sig;
    y_mu_all(NUM_all,:) = y_mu;
    y_sig_all(NUM_all,:)= y_sig;                   	
    % validation set
    vaild_x_feature_label_norm = (vaild_x_feature_label - x_mu) ./ x_sig;
    vaild_y_feature_label_norm = (vaild_y_feature_label - y_mu) ./ y_sig;
    % testing set	
    test_x_feature_label_norm = (test_x_feature_label - x_mu) ./ x_sig;
    test_y_feature_label_norm = (test_y_feature_label - y_mu) ./ y_sig;
    
    y_train_predict_norm=zeros(size(train_y_feature_label,1),size(train_y_feature_label,2));y_vaild_predict_norm=zeros(size(vaild_y_feature_label,1),size(vaild_y_feature_label,2));	
    y_test_predict_norm=zeros(size(test_y_feature_label,1),size(test_y_feature_label,2));	
    for N1=1:length(list_cell)	
        hidden_size = 64;	% hidden_size
        p_train1=cell(size(train_x_feature_label,1),1);p_test1=cell(size(test_x_feature_label,1),1);p_vaild1=cell(size(vaild_x_feature_label,1),1);	
        O_train1=cell(size(train_x_feature_label,1),1);O_test1=cell(size(test_x_feature_label,1),1);O_vaild1=cell(size(vaild_x_feature_label,1),1);	
        for i = 1: size(train_x_feature_label,1)                           
            p_train1{i, 1} = (train_x_feature_label_norm(i,:))';	
        end	
        for i = 1 : size(test_x_feature_label,1)	
            p_test1{i, 1}  = (test_x_feature_label_norm(i,:))';	
        end	
        for i = 1 : size(vaild_x_feature_label,1)	
            p_vaild1{i, 1}  = (vaild_x_feature_label_norm(i,:))';	
        end	
        
        for i = 1: size(train_x_feature_label,1)
            O_train1{i, 1} = (train_y_feature_label_norm(i,list_cell{1,N1}))';	
        end  	
        for i = 1 : size(test_x_feature_label,1)	
            O_test1{i, 1}  = (test_y_feature_label_norm(i,list_cell{1,N1}))';	
        end	
        for i = 1 : size(vaild_x_feature_label,1)	
            O_vaild1{i, 1}  = (vaild_y_feature_label_norm(i,list_cell{1,N1}))';	
        end
	    
        switch 'BiLSTM'
            case 'LSTM'
                model_name = 'LSTM';
                if  (length(hidden_size)<2)	
                    layers = [sequenceInputLayer(size(train_x_feature_label,2)) 	
                              lstmLayer(hidden_size(1), 'OutputMode', 'sequence')  % LSTM layer	
                              reluLayer                                            % Relu layer	
                              dropoutLayer(0.2)                                    % dropoutLayer  	
                              fullyConnectedLayer(size(train_y_feature_label(:,list_cell{1,N1}),2))          % fullyConnectedLayer	
                              regressionLayer];	    
                elseif (length(hidden_size)>=2) 	
                        layers = [sequenceInputLayer(size(train_x_feature_label,2))	
                        lstmLayer(hidden_size(1),'OutputMode','sequence')	
                        dropoutLayer(0.2)	
                        lstmLayer(hidden_size(2),'OutputMode','sequence')	
                        dropoutLayer(0.2)	
                        fullyConnectedLayer(size(train_y_feature_label(:,list_cell{1,N1}),2))	
                        regressionLayer]; 	
                end
            case 'BiLSTM'
                model_name = 'BiLSTM';
                if     (length(hidden_size)<2)	
                       layers = [sequenceInputLayer(size(train_x_feature_label,2)) 	
                       bilstmLayer(hidden_size(1), 'OutputMode', 'sequence')      % BiLSTM	
                       reluLayer                                                  % Relu
                       dropoutLayer(0.2)                                          % dropoutLayer
                       fullyConnectedLayer(size(train_y_feature_label(:,list_cell{1,N1}),2))          % fullyConnectedLayerv	
                       regressionLayer];
                elseif (length(hidden_size)>=2) 	
                       layers = [sequenceInputLayer(size(train_x_feature_label,2))	
                       bilstmLayer(hidden_size(1),'OutputMode','sequence')	
                       dropoutLayer(0.2)	
                       bilstmLayer(hidden_size(2),'OutputMode','sequence')	
                       dropoutLayer(0.2)	
                       gruConnectedLayer(size(train_y_feature_label(:,list_cell{1,N1}),2))	
                       regressionLayer]; 	
                end
        end
        options = trainingOptions('adam', ...	
                                  'MaxEpochs',max_epoch, ...	
                                  'MiniBatchSize',min_batchsize,...	
                                  'InitialLearnRate',0.001,...	           % Learning Rate
                                  'ValidationFrequency',20, ...	
                                  'LearnRateSchedule','piecewise', ...	
                                  'LearnRateDropPeriod',125, ...	
                                  'LearnRateDropFactor',0.2, ...	
                                  'Plots','training-progress');	
        Mdl = trainNetwork(p_train1, O_train1, layers, options);	
        y_train_predict_norm1 = predict(Mdl, p_train1,'MiniBatchSize',min_batchsize);	
        y_vaild_predict_norm1 = predict(Mdl, p_vaild1,'MiniBatchSize',min_batchsize);	
        y_test_predict_norm1 =  predict(Mdl, p_test1,'MiniBatchSize',min_batchsize);	
        y_train_predict_norm_roll=[];y_vaild_predict_norm_roll=[];y_test_predict_norm_roll=[];	
        
        for i=1:length(y_train_predict_norm1)	
          y_train_predict_norm_roll(i,:) = (y_train_predict_norm1{i,1}); 	
        end	
        for i=1:length(y_vaild_predict_norm1)	
          y_vaild_predict_norm_roll(i,:) = (y_vaild_predict_norm1{i,1});	
        end	
        for i=1:length(y_test_predict_norm1) 	
          y_test_predict_norm_roll(i,:) = (y_test_predict_norm1{i,1});  	
        end	
        y_train_predict_norm(:,list_cell{1,N1})=y_train_predict_norm_roll; 	
        y_vaild_predict_norm(:,list_cell{1,N1})=y_vaild_predict_norm_roll; 	
        y_test_predict_norm(:,list_cell{1,N1})=y_test_predict_norm_roll; 		
        graph = layerGraph(Mdl.Layers); figure; plot(graph) 	
    end
	
y_train_predict_cell{1,NUM_all} = y_train_predict_norm.*y_sig+y_mu;          
y_vaild_predict_cell{1,NUM_all} = y_vaild_predict_norm.*y_sig+y_mu;	
y_test_predict_cell{1,NUM_all}  = y_test_predict_norm.*y_sig+y_mu;	
end
y_train_predict = 0;
y_vaild_predict = 0;
y_test_predict  = 0;	
for i=1:length(gas_production_cell)	
    y_train_predict=y_train_predict+ y_train_predict_cell{1,i};	
    y_vaild_predict=y_vaild_predict+ y_vaild_predict_cell{1,i};	
    y_test_predict=y_test_predict+ y_test_predict_cell{1,i};	
end	
	
train_y_feature_label = y_feature_label1(index_label(1:train_num),:); 	
vaild_y_feature_label = y_feature_label1(index_label(train_num+1:vaild_num),:);	
test_y_feature_label  = y_feature_label1(index_label(vaild_num+1:end),:);	
%% Print Result
Tvalue = [if_process,' ',model_name];
train_y=train_y_feature_label; 	
train_MAE=sum(sum(abs(y_train_predict-train_y)))/size(train_y,1)/size(train_y,2) ; disp([Tvalue,' Training set MAE：',num2str(train_MAE)])	
train_RMSE=sqrt(sum(sum(((y_train_predict-train_y)).^2))/size(train_y,1)/size(train_y,2)); disp([Tvalue,' Training set RMSE：',num2str(train_RMSE)]) 	
train_R2 = 1 - mean(norm(train_y - y_train_predict)^2 / norm(train_y - mean(train_y))^2);   disp([Tvalue,' Training set R2：',num2str(train_R2)]) 	
disp('************************************************************************************')	
vaild_y=vaild_y_feature_label;	
vaild_MAE=sum(sum(abs(y_vaild_predict-vaild_y)))/size(vaild_y,1)/size(vaild_y,2) ; disp([Tvalue,' Validation set MAE：',num2str(vaild_MAE)])	
vaild_RMSE=sqrt(sum(sum(((y_vaild_predict-vaild_y)).^2))/size(vaild_y,1)/size(vaild_y,2)); disp([Tvalue,' Validation set RMSE：',num2str(vaild_RMSE)]) 	
vaild_R2 = 1 - mean(norm(vaild_y - y_vaild_predict)^2 / norm(vaild_y - mean(vaild_y))^2);   disp([Tvalue,' Validation set R2：',num2str(vaild_R2)]) 	
disp('************************************************************************************')	
test_y=test_y_feature_label;	
test_MAE=sum(sum(abs(y_test_predict-test_y)))/size(test_y,1)/size(test_y,2) ;disp([Tvalue,' Testing set MAE：',num2str(test_MAE)])		
test_RMSE=sqrt(sum(sum(((y_test_predict-test_y)).^2))/size(test_y,1)/size(test_y,2));disp([Tvalue,' Testing set RMSE：',num2str(test_RMSE)]) 	
test_R2 = 1 - mean(norm(test_y - y_test_predict)^2 / norm(test_y - mean(test_y))^2);disp([Tvalue,' Testing set R2：',num2str(test_R2)])
%% Function
function [x_feature_label,y_feature_label]=timeseries_process2(data_select,select_predict_num,num_series)
num_train = length(data_select)-num_series;
for i = 1:num_train-select_predict_num
  timefeaturedata = data_select(i:i+num_series-1);
  feature_select = [];
  net_input(i,:) = [feature_select(:)',timefeaturedata(:)'];
end
for i = 1:num_train-select_predict_num
  timelabel = data_select(i+num_series:i+num_series+select_predict_num-1);
  net_output(i,:) = timelabel(:)';    
end
  net_input2 = net_input;
  x_feature_label = net_input2;
  y_feature_label = net_output;
end