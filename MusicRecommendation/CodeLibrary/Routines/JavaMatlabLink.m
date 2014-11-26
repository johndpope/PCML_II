% Function calls Java class JavaMatlabLink
% format of cmd call:
% trainFile - file of form: userID,itemID,value
% test file - file of the same form, where only first 2 columns matter
% that would be the pairs for which we want to predict - we can't predict
% for everything, too expensive
% Output file - file of the same form, ideally has same set of userIDs and
% itemIDs as test file, but if  we don't know prediction for some, they
% can be omited (they assumed to be 0 here then)
% Other arguments of a format: RecommenderName, Parameter1, Value1,
% Parameter2, Value2, ...
% Recommender is created based on the recommender name, and all parameters
% are fed to it through setParameters method of its interface.
% For further clarification, please see the following class:
% com.predictionmarketing.RecommenderApp.JavaMatlabLink
function [ Ypred ] = JavaMatlabLink( Ytrain, Ytest,...
    trainFile, testFile, cmdArgs)
    system('rm -rf Mahout/RecommenderApp/target/');
    system('mvn compile -f Mahout/RecommenderApp/pom.xml &>/dev/null');
    writeMatrix(Ytrain, trainFile);
    writeMatrix(Ytest, testFile);
    cmdLine = strcat(...
            'mvn exec:java -Dexec.mainClass=',...
            '"com.predictionmarketing.RecommenderApp.JavaMatlabLink" ',...
            ' -Dexec.args="',...
            trainFile,...
            {' '},...
            testFile,...
            {' JavaOutput '},...
            cmdArgs,...
            '" -f Mahout/RecommenderApp/pom.xml &>/dev/null');
    system(cmdLine{1});
    Ypred = readMatrix('JavaOutput', size(Ytest, 1), size(Ytest, 2));
    % Clean up
    cmdLine = strcat({'rm -rf '}, trainFile);
    system(cmdLine{1});
    cmdLine = strcat({'rm -rf '}, testFile);
    system(cmdLine{1});
    system('rm -rf JavaOutput');
end

