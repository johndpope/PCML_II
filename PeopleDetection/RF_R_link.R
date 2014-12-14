x_1 = as.matrix(read.csv('input_1.mat', header = F, sep = ','));
y_1 = as.matrix(read.csv('label_1.mat', header = F, sep = ','));

x_2 = as.matrix(read.csv('input_2.mat', header = F, sep = ','));
y_2 = as.matrix(read.csv('label_2.mat', header = F, sep = ','));

df = data.frame(x = c(x_1[,1], x_2[,1] + 300), 
                y = c(x_1[,2], x_2[,2] + 300), 
                lab = as.factor(c(y_1, y_2)));
require(ggplot2);
sub = subset(df, lab != 0);
sub$lab = as.factor(as.character(sub$lab));
g = ggplot(sub, aes(x, y)) + geom_point(aes(colour = lab));
print(g);

set.seed(1);
idx = sample(1:nrow(sub), nrow(sub), replace = F);
K = 10;
len = floor(length(idx) / K);

out = data.frame(iter = numeric(0), pred = numeric(0), real = numeric(0));

for(i in 1:K) {
  te = idx[((i - 1) * len + 1):(i * len)];
  tr = setdiff(1:length(idx), te);
  rf = randomForest(lab ~ x + y, data = sub[tr,]);
  pr = predict(rf, newdata = sub[te,], type = 'prob')[,2];
  out = rbind(out, 
              data.frame(iter = 1 + rep(i, length(te)), 
                         pred = 1 + pr, 
                         real = as.numeric(sub$lab[te])) - 1);
}

f = file("output.mat", "w");
write.table(file = f, out, sep = ",", row.names = F, col.names = F);