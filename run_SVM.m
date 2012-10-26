
function rtn = run_SVM()

% add path of libsvm
addpath '~/softwares/libsvm-3.12/matlab/'
addpath '../shared_scripts/'

% actual running
%for name={'emotions','yeast','scene','enron','cal500','fp','cancer','medical','toy10','toy50'}
%X=dlmread(sprintf('/fs/group/urenzyme/workspace/data/%s_features',name{1}));
%Y=dlmread(sprintf('/fs/group/urenzyme/workspace/data/%s_targets',name{1}));

% simulate testing
for name={'toy10'}
X=dlmread(sprintf('../shared_scripts/test_data/%s_features',name{1}));
Y=dlmread(sprintf('../shared_scripts/test_data/%s_targets',name{1}));

rand('twister', 0);

%------------
%
% preparing     
%
%------------
% example selection with meaningful features
Xsum=sum(X,2);
X=X(find(Xsum~=0),:);
Y=Y(find(Xsum~=0),:);
% label selection with two labels
Yuniq=[];
for i=1:size(Y,2)
    if size(unique(Y(:,i)),1)>1
        Yuniq=[Yuniq,i];
    end
end
Y=Y(:,Yuniq);

% feature normalization (tf-idf for text data, scale and centralization for other numerical features)
if or(strcmp(name{1},'medical'),strcmp(name{1},'enron')) 
    X=tfidf(X);
elseif ~(strcmp(name{1}(1:2),'to'))
    X=(X-repmat(min(X),size(X,1),1))./repmat(max(X)-min(X),size(X,1),1);
end

% change Y from -1 to 0: labeling (0/1)
Y(Y==-1)=0;

% length of x and y
Nx = length(X(:,1));
Ny = length(Y(1,:));

% stratified cross validation index
nfold = 3;
Ind = getCVIndex(Y,nfold);

% performance
perf=[];

% get dot product kernels from normalized features or just read precomputed kernels
if or(strcmp(name{1},'fp'),strcmp(name{1},'cancer'))
    K=dlmread(sprintf('/fs/group/urenzyme/workspace/data/%s_kernel',name{1}));
else
    K = X * X'; % dot product
    K = K ./ sqrt(diag(K)*diag(K)');    %normalization diagonal is 1
end

%------------
%
% SVM, single label        
%
%------------
Ypred = [];
YpredVal = [];
% iterate on targets (Y1 -> Yx -> Ym)
for i=1:Ny
    % nfold cross validation
    Ycol = [];
    YcolVal = [];
    for k=1:nfold
        Itrain = find(Ind ~= k);
        Itest  = find(Ind == k);
        % training & testing with kernel
        if strcmp(name{1}(1:2),'to')
                svm_c=0.01;
        elseif strcmp(name{1},'cancer')
                svm_c=5;
        else
                svm_c=0.5;
        end
        model = svmtrain(Y(Itrain,i),[(1:numel(Itrain))',K(Itrain,Itrain)],sprintf('-b 1 -q -c %.2f -t 4',svm_c));
        [Ynew,acc,YnewVal] = svmpredict(Y(Itest,k),[(1:numel(Itest))',K(Itest,Itrain)],model,'-b 1');
        [Ynew] = svmpredict(Y(Itest,k),[(1:numel(Itest))',K(Itest,Itrain)],model);
        Ycol = [Ycol;[Ynew,Itest]];
        if size(YnewVal,2)==2
            YcolVal = [YcolVal;[YnewVal(:,abs(model.Label(1,:)-1)+1),Itest]];
        else
            YcolVal = [YcolVal;[zeros(numel(Itest),1),Itest]];
        end
    end
    Ycol = sortrows(Ycol,size(Ycol,2));
    Ypred = [Ypred,Ycol(:,1)];
    YcolVal = sortrows(YcolVal,size(YcolVal,2));
    YpredVal = [YpredVal,YcolVal(:,1)];
end
% performance of svm
[acc,vecacc,pre,rec,f1,auc1,auc2]=get_performance(Y,Ypred,YpredVal);
perf=[perf;[acc,vecacc,pre,rec,f1,auc1,auc2]];perf


%------------
%
% SVM parameter selection
%
%------------
% parameter selection
svm_cs=[0.01,0.1,0.5,1,5,10];
Isel = randsample(1:size(K,2),ceil(size(K,2)*.03));
IselTrain=Isel(1:ceil(numel(Isel)/3*2));
IselTest=Isel(1:ceil(numel(Isel)/3));
selRes=svm_cs*0;
for j=1:numel(svm_cs)
    svm_c=svm_cs(j);
    Ypred = [];
    YpredVal = [];
    for i=1:Ny
            model = svmtrain(Y(IselTrain,i),[(1:numel(IselTrain))',K(IselTrain,IselTrain)],sprintf('-b 1 -q -c %.2f -t 4',svm_c));
            [Ynew,acc,YnewVal] = svmpredict(Y(IselTest,k),[(1:numel(IselTest))',K(IselTest,IselTrain)],model,'-b 1');
            if size(YnewVal,2)==2
                YnewVal=YnewVal(:,abs(model.Label(1,:)-1)+1);
            else
                YnewVal=zeros(numel(IselTest),1);
            end
            Ypred=[Ypred,YnewVal>0.5];
            YpredVal=[YpredVal,YnewVal];
    end
    selRes(j)=sum(sum(Ypred==Y(IselTest,:)));
end
svm_c=svm_cs(find(selRes==max(selRes)));
if numel(svm_c) >1
    svm_c=svm_c(1);
end

pa=[selRes;svm_cs]
dlmwrite(sprintf('../parameters/%s_para',name{1}),pa)

%------------
%
% single label SVM
%
%------------
YpredVal = [];
% iterate on targets (Y1 -> Yx -> Ym)
for i=1:Ny
    % nfold cross validation
    YcolVal = [];
    for k=1:nfold
        Itrain = find(Ind ~= k);
        Itest  = find(Ind == k);
        % training & testing with kernel
        model = svmtrain(Y(Itrain,i),[(1:numel(Itrain))',K(Itrain,Itrain)],sprintf('-b 1 -q -c %.2f -t 4',svm_c));
        [Ynew,acc,YnewVal] = svmpredict(Y(Itest,k),[(1:numel(Itest))',K(Itest,Itrain)],model,'-b 1');
        if size(YnewVal,2)==2
            YnewVal=YnewVal(:,abs(model.Label(1,:)-1)+1);
        else
            YnewVal=zeros(numel(Itest),1);
        end
        YcolVal = [YcolVal;[YnewVal,Itest]];
    end
    YcolVal = sortrows(YcolVal,size(YcolVal,2));
    YpredVal = [YpredVal,YcolVal(:,1)];
end
% performance of svm
[acc,vecacc,pre,rec,f1,auc1,auc1]=get_performance(Y,YpredVal>0.5,YpredVal);
perf=[perf;[acc,vecacc,pre,rec,f1,auc1,auc2]];perf
% save results
dlmwrite(sprintf('../predictions/%s_predValSVM',name{1}),YpredVal)
dlmwrite(sprintf('../predictions/%s_predBinSVM',name{1}),YpredVal>0.5)
dlmwrite(sprintf('../results/%s_perfSVM',name{1}),perf)


%------------
%
% bagging svm
%
%------------
perf=perf(1,:);
% bagging
Nrep=2;
per=1;
rand('twister', 0);
Ybag=Y*0;
YbagVal=Y*0;
perfBagSVM=[];
perfRandSVM=[];
for b=1:Nrep
    YpredVal = [];
    % iterate on targets (Y1 -> Yx -> Ym)
    for i=1:Ny
        % nfold cross validation
        YcolVal = [];
        for k=1:nfold
            Itrain = find(Ind ~= k);
            BagSize=ceil(numel(Itrain)*per);
            Itrain=randsample(Itrain,BagSize);
            Itest  = find(Ind == k);
            model = svmtrain(Y(Itrain,i),[(1:numel(Itrain))',K(Itrain,Itrain)],sprintf('-b 1 -q -c %.2f -t 4',svm_c));
            [Ynew,acc,YnewVal] = svmpredict(Y(Itest,k),[(1:numel(Itest))',K(Itest,Itrain)],model,'-b 1');
            if size(YnewVal,2)==2
                YnewVal=YnewVal(:,abs(model.Label(1,:)-1)+1);
            else
                YnewVal=zeros(numel(Itest),1);
            end
            YcolVal = [YcolVal;[YnewVal,Itest]];
        end
        YcolVal = sortrows(YcolVal,size(YcolVal,2));
        YpredVal = [YpredVal,YcolVal(:,1)];
    end
    % performance of svm
    [acc,vecacc,pre,rec,f1,auc1,auc2]=get_performance(Y,YpredVal>0.5,YpredVal);
    perfRandSVM=[perfRandSVM;[acc,vecacc,pre,rec,f1,auc1,auc2]];
   
    % performance of bagging
    YbagVal=YbagVal+YpredVal;
    [acc,vecacc,pre,rec,f1,auc1,auc2]=get_performance(Y,YbagVal/b>0.5,YbagVal/b);
    perfBagSVM=[perfBagSVM;[acc,vecacc,pre,rec,f1,auc1,auc2]];
end
perf=[perf;[acc,vecacc,pre,rec,f1,auc1,auc2]];perf

dlmwrite(sprintf('../predictions/%s_predValBagSVM',name{1}),YbagVal/b>0.5)
dlmwrite(sprintf('../predictions/%s_predBinBagSVM',name{1}),YbagVal/b)

% save results
dlmwrite(sprintf('../results/%s_perfBagSVM',name{1}),perf)
dlmwrite(sprintf('../results/%s_perfRandSVM',name{1}),perfRandSVM)
dlmwrite(sprintf('../results/%s_perfBagSVMProce',name{1}),perfBagSVM)

% plot data with true labels
hFig = figure('visible','off');
set(hFig, 'Position', [500,500,1200,50])
subplot(1,5,1);plot(perfBagSVM(:,1));title('Bin accuracy');
subplot(1,5,2);plot(perfBagSVM(:,2));title('multilabel accuracy');
subplot(1,5,3);plot(perfBagSVM(:,5));title('F1');
subplot(1,5,4);plot(perfBagSVM(:,6));title('AUC1');
subplot(1,5,5);plot(perfBagSVM(:,7));title('AUC2');
print(hFig, '-depsc',sprintf('../plots/%s_BagSVM.eps',name{1}));




end
rtn = [];
end



