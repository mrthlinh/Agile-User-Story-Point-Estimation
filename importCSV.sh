echo "Please create database name mydb using <use mydb> and create collection named storypoint using <db.createCollection("storypoint")"
for entry in dataset/*.csv
do
  echo $entry
  # coll=$(echo "$entry" | cut -f 1 -d '.')
  # -d: delimiter, -f: like index of array
  coll=$(echo "$entry" | cut -f 2 -d '/' | cut -f 1 -d '.')
  echo $coll
  # coll=$(echo "$coll" | cut -f 1 -d '.')
  mongoimport -d mydb -c storypoint --type CSV --file $entry --headerline
done
echo "end of import."
