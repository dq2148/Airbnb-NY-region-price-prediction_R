Appendix: R Code
library(randomForest)
library(lattice)
library(caret)
library(tidyr)
library(leaps)

analysisData <- read_csv("~/Desktop/APAN5200/analysisData.csv")
scoringData <- read_csv("~/Desktop/APAN5200/scoringData.csv")

#data clean, remove all variables with words another irrelevant variables
aalysisData1 = subset(analysisData, select = c(host_listings_couht, accommodates, bedrooms, beds, price, security_deposit,
                                                guests_included, cleaning_fee, minimum_nights, maimum_nights, number_of_reviews,
                                                number_of_reviews_ltm, extra_people, review_scores_location, review_scores_value,
                                                review_scores_rating, review_scores_review_scores_accuracy, reviews_per_month,
                                                availability_30, availability_60, availability_90, availability_365,
                                                calculated_host_listings_count, calculated_host_listings_count_etire_homes,
                                                calculated_host_listing_count_private_rooms, calculated_host_listing_cout_shared_rooms,
                                                neighbourhood_group_cleansed, room_type, property_type, cancellation_policy)

#check for missing values and adjust them
sum(is.na(analysisData1$host_listings_count))
sum(is.na(analysisData1$accommodates))
sum(is.na(analysisData1$bathrooms))
sum(is.na(analysisData1$bedrooms))
sum(is.na(analysisData1$beds))
sum(is.na(analysisData1$price))
sum(is.na(analysisData1$security_deposit))
sum(is.na(analysisData1$cleaning_fee))
sum(is.na(analysisData1$guests_included))
sum(is.na(analysisData1$extra_people))
sum(is.na(analysisData1$minimum_nights))
sum(is.na(analysisData1$maximum_nights))
sum(is.na(analysisData1$number_of_reviews))
sum(is.na(analysisData1$number_of_reviews_ltm))
sum(is.na(analysisData1$review_scores_location))
sum(is.na(analysisData1$review_scores_value))
sum(is.na(analysisData1$review_scores_rating))
sum(is.na(analysisData1$review_scores_accuracy))
sum(is.na(analysisData1$availability_30))
sum(is.na(analysisData1$availability_60))
sum(is.na(analysisData1$availability_90))
sum(is.na(analysisData1$availability_365))
sum(is.na(analysisData1$calculated_host_listings_count))
sum(is.na(analysisData1$calculated_host_listings_count_entire_homes))
sum(is.na(analysisData1$calculated_host_listings_count_private_rooms))
sum(is.na(analysisData1$calculated_host_listings_count_shared_rooms))
sum(is.na(analysisData1$reviews_per_month))

#host_listings_count
analysisData1$host_listings_count[which(is.na(analysisData1$host_listings_count))] =
  mean(analysisData1$host_listings_count,na.rm = TRUE)
scoringData$host_listings_count[which(is.na(scoringData$host_listings_count))] =
  mean(scoringData$host_listings_count,na.rm = TRUE)

#beds
analysisData1$beds[which(is.na(analysisData1$beds))] =
  mean(analysisData1$beds,na.rm = TRUE)
scoringData$beds[which(is.na(scoringData$beds))] =
  mean(scoringData$beds,na.rm = TRUE)

#security
analysisData1$security_deposit[which(is.na(analysisData1$security_deposit))] = 0
scoringData$security_deposit[which(is.na(scoringData$security_deposit))] = 0

#cleaning_fee
analysisData1$cleaning_fee[which(is.na(analysisData1$cleaning_fee))] =
  mean(analysisData1$cleaning_fee,na.rm = TRUE)
scoringData$cleaning_fee[which(is.na(scoringData$cleaning_fee))] =
  mean(scoringData$cleaning_fee,na.rm = TRUE)

#reviews_per_month
analysisData1$reviews_per_month[which(is.na(analysisData1$reviews_per_month))] =
  mean(analysisData1$reviews_per_month,na.rm = TRUE)
scoringData$reviews_per_month[which(is.na(scoringData$reviews_per_month))] =
  mean(scoringData$reviews_per_month,na.rm = TRUE)

set.seed(100)
train <- sample(nrow(data), 0.8*nrow(data), replace = FALSE) ##try small
ScoringData <- data[-train,]
dim(analysisData1)

#change factors varuables
analysisData1$cancellation_policy = as.factor(analysisData1$cancellation_policy)
analysisData1$neighbourhood_group_cleansed = as.factor(analysisData1$neighbourhood_group_cleansed)
analysisData1$room_type = as.factor(analysisData1$room_type)
analysisData1$property_type = as.factor(analysisData1$property_type)

#Feature Selection
#Forward Selection
start_mod = lm(price~1,data=analysisData1)
empty_mod = lm(price~1,data=analysisData1)
full_mod = lm(price~.,data=analysisData1)
forwardStepwise = step(start_mod,
                       scope=list(upper=full_mod,lower=empty_mod),
                       direction='forward')

#random forest
test_model <- randomForest(host_listings_couht, accommodates, bedrooms, beds, price, security_deposit,
                       guests_included, cleaning_fee, minimum_nights, maimum_nights, number_of_reviews,
                       number_of_reviews_ltm, extra_people, review_scores_location, review_scores_value,
                       review_scores_rating, review_scores_review_scores_accuracy, reviews_per_month,
                       availability_30, availability_60, availability_90, availability_365,
                       calculated_host_listings_count, calculated_host_listings_count_etire_homes,
                       calculated_host_listing_count_private_rooms, calculated_host_listing_cout_shared_rooms,
                       neighbourhood_group_cleansed, room_type, property_type, cancellation_policy,
                       ntree = 400, mtry = 10, data = analysisData1, importance = TRUE)

test_model  
predRForest = predict(test_model,newdata=scoringData)
rmse = sqrt(mean((predRForest-scoringData$price)^2))


trControl=trainControl(method="cv",number=10)
tuneGrid = expand.grid(mtry=1:10)
set.seed(200)
cvForest = train(price~host_listings_count+accommodates+bathrooms+bedrooms+extra_people+availability_30+availability_60+availability_90+availability_365+
                   number_of_reviews+number_of_reviews_ltm+review_scores_rating+review_scores_location+
                   review_scores_value+calculated_host_listings_count_entire_homes+calculated_host_listings_count_private_rooms+
                   calculated_host_listings_count_shared_rooms+reviews_per_month+cleaning_fee+neighbourhood_group_cleansed+
                   room_type+cancellation_policy+security_deposit,data=TrainSet,
                 method="rf",ntree=400,trControl=trControl,tuneGrid=tuneGrid )
cvForest


#submission for prediction
submissionFile = data.frame(id = scoringData$id, price = predRForest)
write.csv(submissionFile, 'sample_submission.csv',row.names = F)



















