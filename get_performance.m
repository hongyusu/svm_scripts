
function [acc,vecacc,pre,rec,f1] = get_performance(Y,Ypred)

    acc = get_accuracy(Y,Ypred);
    vecacc = get_vecaccuracy(Y,Ypred);
    pre=get_precision(Y,Ypred);
    rec=get_recall(Y,Ypred);
    f1=get_f1(Y,Ypred);

end

function [acc] = get_accuracy(Y,Ypred)

    acc=1-sum(sum(abs(Y-Ypred)))/size(Y,1)/size(Y,2);

end

function [vecacc] = get_vecaccuracy(Y,Ypred)

    vecacc=sum(Y-Ypred,2);
    vecacc=sum((vecacc==0))/numel(vecacc);

end

function [f1] = get_f1(Y,Ypred)

    f1=(2*get_precision(Y,Ypred)*get_recall(Y,Ypred))/(get_precision(Y,Ypred)+get_recall(Y,Ypred));

end

function [tp] = get_tp(Y,Ypred)

    tp = Y + Ypred;
    tp=(tp==2);
    tp=sum(sum(tp));
    
end

function [fp] = get_fp(Y,Ypred)

    fp=Y-Ypred;
    fp=(fp==-1);
    fp=sum(sum(fp));

end

function [tn] = get_tn(Y,Ypred)

    tn=Y+Ypred;
    tn=(tn==0);
    tn=sum(sum(tn));

end

function [fn] = get_fn(Y,Ypred)

    fn=Y-Ypred;
    fn=(fn==1);
    fn=sum(sum(fn));

end

function [pre] = get_precision(Y,Ypred)

    pre=(get_tp(Y,Ypred))/(get_tp(Y,Ypred)+get_fp(Y,Ypred));

end

function [rec] = get_recall(Y,Ypred)

    rec=(get_tp(Y,Ypred))/(get_tp(Y,Ypred)+get_fn(Y,Ypred));

end
