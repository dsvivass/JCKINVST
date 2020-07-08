#!/bin/bash


n=1
command=$1
while ! $command && [ $n -le 10 ]; do
  sleep $n
  ((n+=1))
  echo "Intento numero: $n"
done