library(tidyverse)

# Load in nested CV results for each fold and combine into a table
files <- dir(path = "/path/to/",
             pattern = glob2rx("*_f1Tuned*CVResults.csv"))

CV_raw <- tibble::tibble(filename = files) %>%
  dplyr::mutate(file_contents = purrr::map(filename,
                                           ~ read_csv(file.path("/path/to/", .)))) %>%
  tidyr::unnest() %>%
  dplyr::select(Accuracy, Sensitivity, Specificity, PPV, AUC, F1)

CV <- CV_raw %>%
  tidyr::gather(key = Statistic, value = Score)

# Get mean accuracy
mean(CV_raw$Accuracy)
mean(CV_raw$F1)

# Order variables
CV$Statistic <- factor(CV$Statistic,levels = c("Accuracy", "Sensitivity", "Specificity", "PPV", "AUC", "F1"))

# Dot Plot with Mean + 95% CI indicated
ggplot(data= CV, mapping = aes(x=Statistic, y=Score)) + 
  ylim(0,1) +
  geom_dotplot(binaxis='y', stackdir = 'center', dotsize = 0.7) + 
  stat_summary(fun.y = mean, geom = "errorbar", aes(ymax = ..y.., ymin = ..y..),
               width = 0.75, size = 1, linetype = "solid", colour = "red") +  
  stat_summary(fun.data = mean_cl_normal, geom = "errorbar",
               width = (0.75/2), size = 0.5, linetype = "solid", colour = "red") +
  labs(y = "Score", x = "Statistic") +
  theme(text = element_text(size=12), panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"),
        plot.title = element_text(hjust = 0.5))
ggsave("/path/to/Example_NestedCVResults.pdf", width=5, height=4, units="in")
ggsave("/path/to/Example_NestedCVResults.png", width=5, height=4, units="in")

