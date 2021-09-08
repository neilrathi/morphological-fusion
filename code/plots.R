library(tidyverse)
library(rPref) # pareto
library(extrafont) # font embedding
rm(list=ls())
setwd("~/Desktop/morphological-fusion/")

# simple rectangular sum
pareto_area = function(x, y) {
  area = 0
  basefreq <- -min(x)
  maxfreq <- basefreq + max(x)
  for (i in 1:(length(x)-1)) {
    area <- area + y[i]*((basefreq+x[i+1])-(basefreq+x[i]))
  }
  return(area)
}

# correlation permutation test
permutation_test_corr = function(x, y, num_iter = 10000, method="spearman") {
  standard = cor.test(x, y, method=method)[4]$estimate
  results = c()
  for(i in 1:num_iter) {
    y_shuffled = sample(y)
    result = cor.test(x, y_shuffled, method=method)[4]$estimate
    results = c(results, result)
  }
  mean(results >= standard)
}

# tradeoff permutation test
permutation_test_area = function(df, x, y, num_iter = 10000) {
  # skyline preferences
  p <- low(x) * high(y)
  res <- psel(df, p, top = nrow(df))
  # pareto front
  res1 <- res %>% filter(.level == "1")
  res1 <- res1 %>% drop_na()
  res1 <- res1 %>% add_row(logfreq = max(df$logfreq, na.rm = TRUE), avgsurp = max(df$avgsurp, na.rm = TRUE))
  res1 <- res1 %>% add_row(logfreq = min(df$logfreq), avgsurp = 0)
  res1 <- res1[order(res1$logfreq, res1$avgsurp),]
  # empirical AUC
  standard = with(res1, pareto_area(logfreq, avgsurp))
  results = c()
  for(i in 1:num_iter) {
    # shuffle data
    y_shuffled = sample(y)
    df_shuffled <- data.frame(x, y_shuffled)
    # create skyline as above
    p <- low(x) * high(y_shuffled)
    # pareto front
    res <- psel(df_shuffled, p, top = nrow(df_shuffled))
    res1 <- res %>% filter(.level == "1")
    res1 <- res1 %>% drop_na()
    res1 <- res1 %>% add_row(x = max(df_shuffled$x, na.rm = TRUE), y_shuffled = max(df_shuffled$y_shuffled, na.rm = TRUE))
    res1 <- res1 %>% add_row(x = min(df_shuffled$x), y_shuffled = 0)
    res1 <- res1[order(res1$x, res1$y_shuffled),]
    # shuffled AUC
    result = with(res1, pareto_area(x, y_shuffled))
    results = c(results, result)
  }
  # estimated p-value
  mean(results <= standard)
}

# MAIN FIGURE
d = read_tsv("langdata/lang_data.csv")

d %>%
  separate(code_key, into=c("code", "pos"), sep=" ") %>%
  mutate(pos=case_when(
    pos == "Adjectives" ~ "A",
    pos == "Nouns" ~ "N",
    pos == "Verbs" ~ "V",
    TRUE ~ pos
  )) %>%
  unite(code_key, code, pos, sep=" ") %>%
  ggplot(aes(x=reorder(code_key, surprisal, FUN=median), y=surprisal, fill=family)) +
    geom_boxplot(outlier.shape=NA) +
    stat_summary(fun=mean, geom="point", shape=20, size=3, color="black", fill="black") +
    labs(x="", y="Average Fusion (bits)", fill="") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 40, hjust=0.95, vjust=1.2), 
          legend.position="top", 
          legend.text=element_text(size=7)) +
    guides(fill = guide_legend(nrow = 1, byrow = TRUE)) + 
    ylim(0,50)

ggsave("result_plots/main_figure.pdf", width=13, height=4)
embed_fonts("result_plots/main_figure.pdf")

# FUSION -> SMALL PARADIGM
ds = read_tsv("langdata/surp_size.csv") %>%
  separate(key, into=c("pos", "lang"), sep="\\.") %>%
  mutate(avg_surp=surp/size)

with(ds, cor.test(log(size), avg_surp, method="spearman")) # correlation
with(ds, permutation_test_corr(log(size), avg_surp, method="pearson"))

# generate plot
ds %>%
  ggplot(aes(x=log(size), y=avg_surp, label=paste(lang, pos), color=lang, group=1)) +
    geom_hline(yintercept=0, color="black") +
    stat_smooth(method = 'lm', color = "black") +
    geom_text() +
    theme_minimal() +
    labs(x="Log Paradigm Size", y="Average Fusion") +
    guides(color=F)

ggsave("result_plots/size_fusion_plot.pdf", width=5, height=4)
embed_fonts("result_plots/size_fusion_plot.pdf")

# FUSION -> FREQUENCY
df = read_tsv("langdata/freq_surp.csv") %>%
  mutate(avgsurp=totsurp/numforms) %>%
  separate(features, into=c("feature", "lang"), sep="_")

with(df, permutation_test_area(df, logfreq, avgsurp)) # Pareto curve tradeoff

with(df, cor.test(logfreq, avgsurp, method = "spearman")) # correlation
with(df, permutation_test_corr(logfreq, avgsurp))

# generate plot
p <- low(df$logfreq) * high(df$avgsurp)
res <- psel(df, p, top = nrow(df))
res1 <- res %>% filter(.level == "1")
res1 <- res1 %>% drop_na()
res1 <- res1 %>% add_row(logfreq = max(df$logfreq, na.rm = TRUE), avgsurp = max(df$avgsurp, na.rm = TRUE))
res1 <- res1 %>% add_row(logfreq = min(df$logfreq), avgsurp = 0)
res1 <- res1[order(res1$logfreq, res1$avgsurp),]

res %>%
  ggplot(aes(x = logfreq, y = avgsurp, label = paste(feature, lang, sep="_"), color = lang, group = 1)) + 
  geom_hline(yintercept=0, color="black") +
  geom_point(data = res1) +
  geom_text(size = 2) +
  geom_step(data = res1 , aes(x = logfreq, y = avgsurp), direction="hv", color = "black") +
  stat_smooth(method = "lm", color = "black") +
  theme_minimal() + 
  guides(color = F) +
  labs(x = "Log Normalized Frequency", y = "Average Fusion")

ggsave("result_plots/feature_freq_fusion_plot.pdf", width=5, height=4)
embed_fonts("result_plots/feature_freq_fusion_plot.pdf")