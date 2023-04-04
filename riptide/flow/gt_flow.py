from enum import Enum
from typing import Dict, List, Tuple, Union

import pandas as pd
import plotly.graph_objects as go

from riptide.detection.evaluation import Evaluation, Evaluator


class FlowVisualizer:

    statuses = ["CLS", "LOC", "CLL", "DUP", "MIS", "TP", "FN"]

    def __init__(
        self, evaluators: List[Evaluator], coco_annotations: str, img_dir: str
    ):
        assert (
            len(evaluators) >= 2
        ), "At least two evaluators are required to create a flow diagram"
        # TODO: Check that all evaluators are based on the same targets (i.e. same COCO annotations file)
        self.evaluators = evaluators
        self.coco_annotations = coco_annotations
        self.img_dir = img_dir

        self.confusion_matrices: Dict[Tuple[int, int], Dict[str, dict]] = dict()

    def compute_status_flow(
        self, evaluations: List[Tuple[int, Evaluation]]
    ) -> Tuple[List[Dict], List[Dict]]:
        unassigned: List[Dict] = []
        gt_status_flow = []
        for idx, evaluation in evaluations:
            for gt_id, statuses in evaluation.get_status().items():
                if gt_id is None:
                    unassigned.extend(
                        [
                            {"gt_id": gt_id, "idx": idx, **status.todict()}
                            for status in statuses
                        ]
                    )
                    continue
                # take highest confidence status as representative
                status = statuses[0].copy()

                num_tp = status.score if status.state.code == "TP" else 0
                total = sum([s.score for s in statuses])
                status.score = (
                    2 * num_tp / (total + num_tp) if total + num_tp > 0 else 0
                )

                gt_status_flow.append({"gt_id": gt_id, "idx": idx, **status.todict()})
        return unassigned, gt_status_flow

    def generate_graph(
        self,
        ids: List[int] = [0, 1],
        labels: Union[str, int, List[int]] = "all",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if labels == "all":
            evaluations = [
                (i, evaluation)
                for i, id in enumerate(ids)
                for evaluation in self.evaluators[id].evaluations
            ]

        else:
            # TODO: Extract evaluations by labels
            raise NotImplementedError(
                "int and list of int are not currently supported for labels"
            )

        # generate graph nodes
        unassigned, gt_status_flow = self.compute_status_flow(evaluations)

        df = pd.DataFrame.from_records(gt_status_flow).convert_dtypes()
        unassigned_df = pd.DataFrame.from_records(unassigned).convert_dtypes()

        nodes = (
            pd.concat([df, unassigned_df], ignore_index=True)
            .groupby(["idx", "state"])[["score", "iou", "gt_id"]]
            .agg(
                {
                    "score": ["mean", "std", "count"],
                    "iou": ["mean", "std", "count"],
                    "gt_id": set,
                }
            )
            .reset_index()
            .reset_index(names=["node_id"])
            .convert_dtypes()
        )
        nodes.columns = [
            "_".join(pair) if pair[1] != "" else pair[0] for pair in nodes.columns
        ]
        nodes = nodes.rename(columns={"gt_id_set": "gt_ids"})

        # TODO: BKG, UN edges not collated correctly
        edges = pd.merge(
            nodes, nodes, how="cross", suffixes=("_source", "_target")
        ).rename(
            columns={
                "node_id_source": "source",
                "node_id_target": "target",
                "score_count_source": "count",
            }
        )
        edges = edges[
            (edges["idx_source"] < edges["idx_target"])
            & (edges["state_source"] != "UN")
        ]
        edges["gt_ids"] = edges.apply(
            lambda row: row["gt_ids_source"] & row["gt_ids_target"], axis=1
        )
        edges["weight"] = edges["gt_ids"].apply(len)
        edges = edges[edges["weight"] > 0]
        edges["weight"].where(
            edges["gt_ids"].apply(sum).notna(), edges["count"], inplace=True
        )
        edges["score"] = edges["score_mean_target"] - edges["score_mean_source"]
        # edges = edges[edges["score"] > 0]
        edges["score_std"] = (
            edges["score_std_source"] ** 2 + edges["score_std_target"] ** 2
        ) ** (0.5)

        edges = edges[
            [
                "source",
                "target",
                "state_source",
                "state_target",
                "weight",
                "gt_ids",
                "score",
                "score_std",
            ]
        ].reset_index(drop=True)

        nodes.drop(columns=["node_id"], inplace=True)

        return nodes, edges

    class FlowType(str, Enum):
        """Enum to represent the type of flow in the sankey diagram"""

        SANKEY = "sankey"
        SUNBURST = "sunburst"
        CHORD = "chord"
        NETWORK = "network"

    def visualize(
        self,
        title_text: str = "Flow of ground truth statuses",
        ids: Tuple[int, ...] = (0, 1),
        display_type: FlowType = FlowType.SANKEY,
    ) -> go.Figure:
        """Visualize the change in status of ground truths between two models.

        The sankey diagram will be saved as an attribute in the class

        Parameters
        ----------
        title_text : str
            The title of the sankey diagram
        ids : Tuple[int, int], optional
            The ids of the two models that are to be compared, by default (0,1)
        """
        assert len(ids) == 2, "Must provide two ids to compare"

        # compute the graph
        nodes, edges = self.generate_graph(ids, labels="all")
        models = [self.evaluators[i].name for i in ids]

        if display_type is not self.FlowType.SANKEY:
            raise NotImplementedError("Only sankey diagram is supported for now")

        # region: build sankey diagram

        node_labels = [
            f"{models[ids[attr['idx']]]} {attr['state']}"
            for node, attr in nodes.iterrows()
        ]

        ## source represents the indexes of the labels that are to be used
        ## as the starting point of a "sankey flow"
        ## the value in the source array corresponds to
        ## the value in same position in the target array

        ## target represents the indexes of the labels that are to be used
        ## as the ending point of a "sankey flow"
        ## the value in the target array corresponds to
        ## the value in same position in the source array

        source, target = edges[["source", "target"]].values.T.tolist()
        attrs = edges[["weight", "gt_ids", "score"]].to_dict(orient="records")

        edge_labels = [f"{attr['score']} {attr['weight']}" for attr in attrs]

        value = [attr["weight"] for attr in attrs]

        # endregion

        fig = go.Figure(
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.1),
                    label=node_labels,
                    color="blue",
                    # customdata = [list(dictionary.values()) for dictionary in self.confusion_matrix.values()] + [list]
                    # hovertemplate='%{label} breakdown:<br> ',
                ),
                link=dict(
                    source=source,  # indices correspond to labels, eg A1, A2, A1, B1, ...
                    target=target,
                    value=value,
                    label=edge_labels,
                ),
            ),
            layout=dict(
                title_text=title_text,
                font_size=10,
            ),
        )

        return fig
