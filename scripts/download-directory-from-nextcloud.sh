# Only use with source ./download-directory-from-nextcloud.sh from commonroad-geometric/scripts/
# $1, download link from nextcloud directory
# $2, new directory name within data

# Prepare data directories
cd ..
mkdir ./data/ ./data/raw/ ./data/interim ./data/processed
# Download directory as .zip from nextcloud to directory "raw"
curl -X get $1 -o ./data/raw/$2.zip

# Unzip it to directory "interim"
pushd data
unzip ./raw/$2.zip -d ./interim/$2/

pushd interim
pushd $2
# Ignore what the directory within the $2.zip from nextcloud is called and move it into a directory called $2
cd *
for file in ./*; do
  mv "${file##*/}" ../"${file##*/}"
done
cd ..
popd
popd

# Extract everything within the directory (non-recursive) to directory "processed"
# Everything we store on nextcloud should be a .zip/.zstd anyway

# For .zip
for zipfile in ./interim/$2/*.zip; do
    unzip "$zipfile" -d ./processed/$2/
done

# For .tar.zstd
for tarfile in ./interim/$2/*.tar.zstd; do
    tar --use-compress-program=unzstd -xvf "$tarfile" -C ./processed/$2/
done

# Pop data
popd
# Move back into scripts
cd scripts
