FROM ubuntu:18.10

MAINTAINER Joze Rozanec <jmrozanec@gmail.com>

#We take some inspiration from here: https://github.com/vegansk/docker-ubuntu-java-nodejs/blob/master/Dockerfile
RUN apt-get update
RUN apt-get install -y curl git sudo build-essential vim python3-pip 
RUN curl -sL https://deb.nodesource.com/setup_10.x | bash
RUN apt-get install -y nodejs
RUN sudo npm install -g dev-refresh node-dev onchange
RUN curl -sS https://dl.yarnpkg.com/debian/pubkey.gpg | apt-key add -
RUN echo "deb https://dl.yarnpkg.com/debian/ stable main" | tee /etc/apt/sources.list.d/yarn.list
RUN apt-get update
RUN apt-get install -y yarn
RUN apt-get install -y bzip2 gzip zip tar

RUN mkdir -p /var/log/eolearn/

#######################################################################################################################################
ARG user=eolearn
ARG group=eolearn
ARG uid=1000
ARG gid=1000
ARG USER_HOME="/home/${user}"

ENV USER_HOME ${USER_HOME}

RUN groupadd -g ${gid} ${group} && useradd -d "${USER_HOME}" -u "${uid}" -g "${gid}" -m -s /bin/bash "${user}"
RUN echo "eolearn:eolearn" | chpasswd 

# setup SSH server
RUN apt-get update && apt-get install --no-install-recommends -y openssh-server && rm -rf /var/lib/apt/lists/*

RUN sed -i /etc/ssh/sshd_config \
        -e 's/#PermitRootLogin.*/PermitRootLogin no/' \
        -e 's/#RSAAuthentication.*/RSAAuthentication no/'  \
        -e 's/#PasswordAuthentication.*/PasswordAuthentication yes/' \
        -e 's/#SyslogFacility.*/SyslogFacility AUTH/' \
        -e 's/#LogLevel.*/LogLevel INFO/' && \
    mkdir /var/run/sshd

# SSH login fix. Otherwise user is kicked off after login
# https://docs.docker.com/engine/examples/running_ssh_service/
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

WORKDIR "${USER_HOME}"

ADD ./ "${USER_HOME}/"
COPY setup-sshd /usr/local/bin/setup-sshd

EXPOSE 22

ENTRYPOINT ["setup-sshd"]
