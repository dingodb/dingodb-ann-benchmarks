# -*- coding: UTF-8 -*-
import os
import dingodb
import numpy
from ..base.module import BaseANN

ann_host = os.environ.get("ANN_HOST")
ann_port = os.environ.get("ANN_PORT")
connect_user = os.environ.get("ConnectUser")
connect_pass = os.environ.get("ConnectPass")
host = ann_host + ":" + ann_port

default_server_config = {
    "user": connect_user,
    "password": connect_pass,
    "host": [host]
}


def metric_mapping(_metric: str):
    # _metric_type = {"angular": "cosine", "euclidean": "l2"}.get(_metric, None)
    _metric_type = {"angular": "dotproduct", "euclidean": "euclidean"}.get(_metric, None)
    if _metric_type is None:
        raise Exception(f"[Dingodb] Not support metric type: {_metric}!!!")
    return _metric_type


class DingoDB(BaseANN):

    def __init__(self, metric, index_param):
        self._server_param = default_server_config
        self._metric = metric
        # print(metric)
        self._metric_type = metric_mapping(self._metric)
        self.index_param = index_param
        self.param_string = "-".join(k + "-" + str(v) for k, v in self.index_param.items()).lower()
        self.index_name = f"demo-{self.param_string}"
        # self._index_m = index_param.get("M", None)
        self._index_m = index_param["M"]
        # self._index_ef = index_param.get("efConstruction", None)
        self._index_ef = index_param["efConstruction"]
        self._search_ef = None
        self.client = self.getClientStatus()

    def getClientStatus(self):
        dingodb_client = dingodb.DingoDB(self._server_param.get("user"), self._server_param.get("password"),
                                         self._server_param.get("host"))
        if not dingodb_client:
            print("Failed to connect to Dingodb")
        else:
            return dingodb_client

    def fit(self, X) -> None:
        dim = X.shape[1]
        index_list = self.client.get_index()

        if self.index_name in index_list:
            self.client.delete_index(self.index_name)

        index_conf = {
            "efConstruction": self._index_ef,
            "maxElements": len(X),
            "nlinks": self._index_m
        }

        self.index = self.client.create_index(index_name=self.index_name, dimension=dim, metric_type=self._metric_type,
                                              index_config=index_conf)

        if X.dtype != numpy.float32:
            X = X.astype(numpy.float32)

        vector_num = 0
        for _ in X:
            if vector_num % 1000 == 0:
                self.client.vector_add(self.index_name, None, X[vector_num: vector_num + 1000].tolist())
            vector_num += 1

    def set_query_arguments(self, ef):
        self._search_ef = ef

    def query(self, v, n):
        # res = self.client.vector_search(self.index_name, v.tolist(), top_k=n,
        #                                 search_params={"efSearch": self._search_ef})

        res = self.client.vector_search(self.index_name, v.tolist(), top_k=n,
                                        search_params={
                                            "efSearch": self._search_ef,
                                            "withScalarData": "false",
                                            "withoutVectorData": "true"
                                        })

        # print(numpy.array([vector['id'] for item in res for vector in item["vectorWithDistances"]]))
        self.client.close()

        return numpy.array([vector['id'] - 1 for item in res for vector in item["vectorWithDistances"]])

    def __str__(self):
        return f"DingoDB (index_M:{self._index_m},index_ef:{self._index_ef}, search_ef={self._search_ef})"
