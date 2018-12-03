for entry in dataset/*.csv
do
  echo $entry
  # coll=$(echo "$entry" | cut -f 1 -d '.')
  # -d: delimiter, -f: like index of array
  coll=$(echo "$entry" | cut -f 2 -d '/' | cut -f 1 -d '.')
  # coll=$(echo "$coll" | cut -f 1 -d '.')
  mongoimport -d mydb -c storypoint --type CSV --file $coll".csv" --headerline
done
echo "end of import."
