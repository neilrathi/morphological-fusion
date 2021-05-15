rm(list=ls())
setwd("~/projects/morphological-fusion/")
library(tidyverse)

permutation_test = function(x, y, num_iter=10000, method="spearman") {
  standard = cor.test(x, y, method=method)[4]$estimate
  results = c()
  for(i in 1:num_iter) {
    y_shuffled = sample(y)
    result = cor.test(x, y_shuffled, method=method)[4]$estimate
    results = c(results, result)
  }
  mean(results >= standard)
}

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

ds = read_tsv("langdata/surp_size.csv") %>%
  separate(key, into=c("pos", "lang"), sep="\\.") %>%
  mutate(avg_surp=surp/size)

# FUSION -> SMALL PARADIGM

ds %>%
  ggplot(aes(x=log(size), y=avg_surp, label=paste(lang, pos), color=lang, group=1)) +
    geom_hline(yintercept=0, color="black") +
    stat_smooth(method='lm') +
    geom_text() +
    theme_minimal() +
    labs(x="Log Paradigm Size", y="Average Fusion") +
    guides(color=F)

ggsave("result_plots/size_fusion_plot.pdf", width=5, height=4)


with(ds, cor.test(log(size), avg_surp, method="spearman"))
with(ds, permutation_test(log(size), avg_surp, method="pearson"))


# FUSION -> FREQUENCY

df = read_tsv("langdata/freq_surp.csv") %>%
  mutate(avgsurp=totsurp/numforms) %>%
  separate(features, into=c("feature", "lang"), sep="_")

df %>%
  ggplot(aes(x=logfreq, y=avgsurp, label=paste(feature, lang, sep="_"), color=lang, group=1)) +
  geom_hline(yintercept=0, color="black") +
  stat_smooth(method='lm') +
  geom_text(size=2) +
  theme_minimal() +
  guides(color=F) +
  labs(x="Log Normalized Frequency", y="Average Fusion")

ggsave("result_plots/feature_freq_fusion_plot.pdf", width=5, height=4)

# Without Hungarian:

df %>%
  filter(lang != "hun") %>%
  ggplot(aes(x=logfreq, y=avgsurp, label=paste(feature, lang, sep="_"), color=lang, group=1)) +
  stat_smooth(method='lm') +
  geom_text(size=2) +
  theme_minimal() +
  guides(color=F) +
  labs(x="Log Frequency", y="Average Fusion")

with(df, cor.test(logfreq, avgsurp, method="pearson"))
with(df, cor.test(logfreq, avgsurp, method="spearman"))

df %>%
  filter(lang != "hun") %>% 
  with(cor.test(logfreq, avgsurp, method="spearman")) # it's even stronger

with(df, permutation_test(logfreq, avgsurp))


