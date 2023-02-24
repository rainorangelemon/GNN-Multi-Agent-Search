#!/bin/bash
ListNegativeSampling="around_optimal just_optimal"
ListTrueFalse="False True"

ArrNegativeSampling=($ListNegativeSampling)
ArrTrueFalse=($ListTrueFalse)


# user="USER INPUT"
read -p "Enter negative sampling (0 - around_optimal; 1 - just_optimal): " IdNegativeSampling

echo ${ArrNegativeSampling[$IdNegativeSampling]}