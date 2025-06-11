from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.lib import hub
import asyncio

import switch
from datetime import datetime


import math

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

class SimpleMonitor13(switch.SimpleSwitch13):

    def __init__(self, *args, **kwargs):
        #print("in init function")
        super(SimpleMonitor13, self).__init__(*args, **kwargs)
        self.datapaths = {}
        self.monitor_thread = hub.spawn(self._monitor)

        start = datetime.now()

        self.flow_training()

        end = datetime.now()
        print("Training time: ", (end-start))

    @set_ev_cls(ofp_event.EventOFPStateChange,
                [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change_handler(self, ev):
        #print("In state change function")
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            if datapath.id not in self.datapaths:
                self.logger.debug('register datapath: %016x', datapath.id)
                self.datapaths[datapath.id] = datapath
        elif ev.state == DEAD_DISPATCHER:
            if datapath.id in self.datapaths:
                self.logger.debug('unregister datapath: %016x', datapath.id)
                del self.datapaths[datapath.id]

    def _monitor(self):
        #print("in monitor function")
        while True:
            for dp in self.datapaths.values():
                self._request_stats(dp)
            hub.sleep(10)
           
           
            asyncio.run(self.flow_predict())
   
   
    def _request_stats(self, datapath):
        #print("in request stats function")
        self.logger.debug('send stats request: %016x', datapath.id)
        parser = datapath.ofproto_parser

        req = parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply_handler(self, ev):
        #print("in flow stats reply handler function")
        timestamp = datetime.now()
        timestamp = timestamp.timestamp()

        file0 = open("PredictFlowStatsfile.csv","w")
        file0.write('timestamp,datapath_id,flow_id,ip_src,tp_src,ip_dst,tp_dst,ip_proto,icmp_code,icmp_type,flow_duration_sec,flow_duration_nsec,idle_timeout,hard_timeout,flags,packet_count,byte_count,packet_count_per_second,packet_count_per_nsecond,byte_count_per_second,byte_count_per_nsecond\n')
        body = ev.msg.body
        icmp_code = -1
        icmp_type = -1
        tp_src = 0
        tp_dst = 0

        for stat in sorted([flow for flow in body if (flow.priority == 1) ], key=lambda flow:
            (flow.match['eth_type'],flow.match['ipv4_src'],flow.match['ipv4_dst'],flow.match['ip_proto'])):
       
            ip_src = stat.match['ipv4_src']
            ip_dst = stat.match['ipv4_dst']
            ip_proto = stat.match['ip_proto']
           
            if stat.match['ip_proto'] == 1:
                icmp_code = stat.match['icmpv4_code']
                icmp_type = stat.match['icmpv4_type']
               
            elif stat.match['ip_proto'] == 6:
                tp_src = stat.match['tcp_src']
                tp_dst = stat.match['tcp_dst']

            elif stat.match['ip_proto'] == 17:
                tp_src = stat.match['udp_src']
                tp_dst = stat.match['udp_dst']

            flow_id = str(ip_src) + str(tp_src) + str(ip_dst) + str(tp_dst) + str(ip_proto)
         
            try:
                packet_count_per_second = stat.packet_count/stat.duration_sec
                packet_count_per_nsecond = stat.packet_count/stat.duration_nsec
            except:
                packet_count_per_second = 0
                packet_count_per_nsecond = 0
               
            try:
                byte_count_per_second = stat.byte_count/stat.duration_sec
                byte_count_per_nsecond = stat.byte_count/stat.duration_nsec
            except:
                byte_count_per_second = 0
                byte_count_per_nsecond = 0
               
            file0.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n"
                .format(timestamp, ev.msg.datapath.id, flow_id, ip_src, tp_src,ip_dst, tp_dst,
                        stat.match['ip_proto'],icmp_code,icmp_type,
                        stat.duration_sec, stat.duration_nsec,
                        stat.idle_timeout, stat.hard_timeout,
                        stat.flags, stat.packet_count,stat.byte_count,
                        packet_count_per_second,packet_count_per_nsecond,
                        byte_count_per_second,byte_count_per_nsecond))
           
        file0.close()

    def flow_training(self):
        #print("in flow training function")
        self.logger.info("Flow Training ...")

        flow_dataset = pd.read_csv('FlowStatsfile.csv')

        flow_dataset.iloc[:, 2] = flow_dataset.iloc[:, 2].str.replace('.', '')
        flow_dataset.iloc[:, 3] = flow_dataset.iloc[:, 3].str.replace('.', '')
        flow_dataset.iloc[:, 5] = flow_dataset.iloc[:, 5].str.replace('.', '')

        X_flow = flow_dataset.iloc[:, :-1].values
        X_flow = X_flow.astype('float64')

        y_flow = flow_dataset.iloc[:, -1].values

        X_flow_train, X_flow_test, y_flow_train, y_flow_test = train_test_split(X_flow, y_flow, test_size=0.25, random_state=0)
        ###############################
        DT_classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
        self.DT_flow_model = DT_classifier.fit(X_flow_train, y_flow_train)
        ###############################
        RF_classifier = RandomForestClassifier(n_estimators=10, criterion="entropy", random_state=0)
        self.RF_flow_model = RF_classifier.fit(X_flow_train, y_flow_train)
        ###############################
       
        LR_classifier = LogisticRegression(solver='liblinear', random_state=0)
        self.LR_flow_model=LR_classifier.fit(X_flow_train, y_flow_train)
        ###############################
        y_flow_pred = self.DT_flow_model.predict(X_flow_test)

        self.logger.info("-----------------------DECISION TREE-----------------------------------------")

        self.logger.info("confusion matrix")
        cm = confusion_matrix(y_flow_test, y_flow_pred)
        self.logger.info(cm)

        acc = accuracy_score(y_flow_test, y_flow_pred)

        self.logger.info("succes accuracy = {0:.2f} %".format(acc*100))
        fail = 1.0 - acc
        self.logger.info("fail accuracy = {0:.2f} %".format(fail*100))
        self.logger.info("------------------------------------------------------------------------------")
        ################################
       
        y_flow_pred = self.RF_flow_model.predict(X_flow_test)

        self.logger.info("-------------------------RANDOM FOREST----------------------------------------")

        self.logger.info("confusion matrix")
        cm = confusion_matrix(y_flow_test, y_flow_pred)
        self.logger.info(cm)

        acc = accuracy_score(y_flow_test, y_flow_pred)

        self.logger.info("succes accuracy = {0:.2f} %".format(acc*100))
        fail = 1.0 - acc
        self.logger.info("fail accuracy = {0:.2f} %".format(fail*100))
        self.logger.info("------------------------------------------------------------------------------")
       
        ###################################
       
        y_flow_pred = self.LR_flow_model.predict(X_flow_test)

        self.logger.info("-----------------------LOGISTIC REGRESSION-----------------------------------------")

        self.logger.info("confusion matrix")
        cm = confusion_matrix(y_flow_test, y_flow_pred)
        self.logger.info(cm)

        acc = accuracy_score(y_flow_test, y_flow_pred)

        self.logger.info("succes accuracy = {0:.2f} %".format(acc*100))
        fail = 1.0 - acc
        self.logger.info("fail accuracy = {0:.2f} %".format(fail*100))
        self.logger.info("------------------------------------------------------------------------------")      
       
    async def flow_predict(self):
        #print("in flow _predict function")
        try:
            predict_flow_dataset = pd.read_csv('PredictFlowStatsfile.csv')
            rows_count = len(predict_flow_dataset.index)
           
            #Entropy based detection
            print("...")
            print("...")
            print("Entropy Based Detection")
            print("...")
            print("...")
            dst_input=list(predict_flow_dataset.loc[:,'ip_dst'])
           
           
            length = len(dst_input)
            i=0
            async def calculateEntropy(ip_dst):
               
                self.pktCnt += 1
                if ip_dst in self.ipList_Dict:
                    self.ipList_Dict[ip_dst] += 1
                else:
                    self.ipList_Dict[ip_dst] = 0
               
                if self.pktCnt == 5:
                    self.sumEntropy = 0
                    self.ddosDetected = 0
                   
                    for ip_dst,value in self.ipList_Dict.items():
                        prob = abs(value/float(self.pktCnt))
                        if (prob > 0.0):
                            ent = -prob * math.log(prob,2)
                            self.sumEntropy = self.sumEntropy + ent
                   
                    if (self.sumEntropy < 2 and self.sumEntropy != 0):
                        self.counter += 1
                    else:
                        self.counter = 0
                    if self.counter == 10:
                        self.ddosDetected = 1
                        #print("Counter = ", self.counter)
                       
                        if(self.ddosDetected == 1):
                            #ML Model
                            print("ENTROPY HAS DETECTED DDOS ATTACK PASSING THE TRAFFIC TO ML MODULE")
                            votes=[]
                            print("DECISION TREE ML Model")
                            predict_flow_dataset.iloc[:, 2] = predict_flow_dataset.iloc[:, 2].str.replace('.', '')
                            predict_flow_dataset.iloc[:, 3] = predict_flow_dataset.iloc[:, 3].str.replace('.', '')
                            predict_flow_dataset.iloc[:, 5] = predict_flow_dataset.iloc[:, 5].str.replace('.', '')

                            X_predict_flow = predict_flow_dataset.iloc[:, :].values
                            X_predict_flow = X_predict_flow.astype('float64')
                            #################################################################################
                            y_flow_pred = self.DT_flow_model.predict(X_predict_flow)
                           
                            legitimate_trafic = 0
                            ddos_trafic = 0

                            for i in y_flow_pred:
                                if i == 0:
                                    legitimate_trafic = legitimate_trafic + 1
                                else:
                                    ddos_trafic = ddos_trafic + 1
                                    victim = int(predict_flow_dataset.iloc[i, 5])%20
                   
                   
                   

                            self.logger.info("------------------------------------------------------------------------------")
                            if (legitimate_trafic/len(y_flow_pred)*100) > 80:
                               
                                votes.append(0)
                                self.logger.info("DECISION TREE votes as legitimate trafic ...")
                            else:
                                votes.append(1)
                                self.logger.info("DECISION TREE votes as ddos trafic ...")
                                self.logger.info("victim is host: h{}".format(victim))
                               
                               
                            self.logger.info("------------------------------------------------------------------------------")
           
                           
                            ###################################################################################
                            print("RANDOM FOREST ML Model")
                            y_flow_pred = self.RF_flow_model.predict(X_predict_flow)
                           
                            legitimate_trafic = 0
                            ddos_trafic = 0

                            for i in y_flow_pred:
                                if i == 0:
                                    legitimate_trafic = legitimate_trafic + 1
                                else:
                                    ddos_trafic = ddos_trafic + 1
                                    victim = int(predict_flow_dataset.iloc[i, 5])%20
                   
                   
                   

                            self.logger.info("------------------------------------------------------------------------------")
                            if (legitimate_trafic/len(y_flow_pred)*100) > 80:
                                votes.append(0)
                                self.logger.info("RandomForest considers votes as legitimate trafic ...")
                            else:
                                votes.append(1)
                                self.logger.info("RandomForest considers votes as ddos trafic ...")
                                self.logger.info("victim is host: h{}".format(victim))

                            self.logger.info("------------------------------------------------------------------------------")
                            ########################################################################################
                           
                           
                            y_flow_pred = self.LR_flow_model.predict(X_predict_flow)
                           
                            legitimate_trafic = 0
                            ddos_trafic = 0

                            for i in y_flow_pred:
                                if i == 0:
                                    legitimate_trafic = legitimate_trafic + 1
                                else:
                                    ddos_trafic = ddos_trafic + 1
                                    victim = int(predict_flow_dataset.iloc[i, 5])%20
                   
                   
                   

                            self.logger.info("------------------------------------------------------------------------------")
                            if (legitimate_trafic/len(y_flow_pred)*100) > 80:
                                votes.append(0)
                                self.logger.info("Logistic Regression votes to legitimate trafic ...")
                            else:
                                votes.append(1)
                                self.logger.info("Logistic Regression votes to ddos trafic ...")
                                self.logger.info("victim is host: h{}".format(victim))

                            self.logger.info("------------------------------------------------------------------------------")
                            ###########################################################################################
                           
                            counts = {}
                            for vote in votes:
                                if vote in counts:
                                    counts[vote] += 1
                                else:
                                    counts[vote] = 1

                            max_count = 0
                            max_vote = None
                            for vote, count in counts.items():
                                if count > max_count:
                                    max_count = count
                                    max_vote = vote
                            if max_vote==1:
                                print("Hardly voted by models is the traffic is DDOS ")
                            else:
                                print("Hardly voted by models is the traffic is legitimate ")
                           
                            ###########################################################################################
                            file0 = open("PredictFlowStatsfile.csv","w")
           
                            file0.write('timestamp,datapath_id,flow_id,ip_src,tp_src,ip_dst,tp_dst,ip_proto,icmp_code,icmp_type,flow_duration_sec,flow_duration_nsec,idle_timeout,hard_timeout,flags,packet_count,byte_count,packet_count_per_second,packet_count_per_nsecond,byte_count_per_second,byte_count_per_nsecond\n')
                            file0.close()
                        else:
                            pass
                        self.counter = 0
                    self.pktCnt = 0
                    self.dst_ipList = []
                    self.ipList_Dict = {}    
           
                   
                       
            self.pktCnt = 0
            self.ddosDetected = 0
            self.counter = 0
            self.ipList_Dict = {}
            self.sumEntropy = 0
                 
            while i< length:
               
                ip_dst = dst_input[i]
               
                await calculateEntropy(ip_dst)
                if(self.ddosDetected == 1):
                    i = length +1
               
                i += 1

        except Exception as e:
            print(e)
