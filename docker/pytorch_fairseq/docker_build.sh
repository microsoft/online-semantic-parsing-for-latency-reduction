# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
docker_user=jzhou316
image_name=pytorch-fairseq
tag_name=py3.8.10-torch1.8.1-cuda10.2-fairseq7ca8bc1
docker build -t $docker_user/$image_name:$tag_name .

# push to docker hub
docker push $docker_user/$image_name:$tag_name
