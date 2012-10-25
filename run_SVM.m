
function rtn = run_SVM()

% add path of libsvm
addpath '~/softwares/libsvm-3.12/matlab/'


% actual running
%for name={'emotions','yeast','scene','enron','cal500','fp','cancer','medical','toy10','toy50'}
%X=dlmread(sprintf('/fs/group/urenzyme/workspace/data/%s_features',name{1}));
%Y=dlmread(sprintf('/fs/group/urenzyme/workspace/data/%s_targets',name{1}));

% simulate testing
for name={'toy10'}
X=dlmread(sprintf('./test_data/%s_features',name{1}));
Y=dlmread(sprintf('./test_data/%s_targets',name{1}));

rand('twister', 0);

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
                svm_c=5
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
[ax,ay,t,auc]=perfcurve(reshape(Y,1,numel(Y)),reshape(YpredVal,1,numel(Y)),1);
auc1=get_auc(Y,YpredVal);
[acc,vecacc,pre,rec,f1]=get_performance(Y,Ypred);
perf=[perf;[acc,vecacc,pre,rec,f1,auc,auc1]];perf


%------------
%
% SVM
%
%------------
% parameter selection
svm_cs=[0.01,0.1,1,5,10];
Isel = randsample(1:size(K,2),ceil(size(K,2)*.05));
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

selRes
svm_c

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
[ax,ay,t,auc]=perfcurve(reshape(Y,1,numel(Y)),reshape(YpredVal,1,numel(Y)),1);
auc1=get_auc(Y,YpredVal);
[acc,vecacc,pre,rec,f1]=get_performance(Y,Ypred);
perf=[perf;[acc,vecacc,pre,rec,f1,auc,auc1]];perf


dlmwrite(sprintf('../predictions/%s_predValSVM',name{1}),YpredVal)
dlmwrite(sprintf('../predictions/%s_predBinSVM',name{1}),YpredVal>=0.5)

% save results
dlmwrite(sprintf('../results/%s_perfSVM',name{1}),perf)
end

rtn = [];
end




function [auc] = get_auc(Y,YpredVal)
    AUC=zeros(1,size(Y,2));
    for i=1:size(Y,2)
        [ax,ay,t,AUC(1,i)]=perfcurve(Y(:,i),YpredVal(:,i),1);
    end
    auc=mean(AUC);
end


