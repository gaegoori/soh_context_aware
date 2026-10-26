from pathlib import Path
import xml.etree.ElementTree as ET
from drawing_utils import Diagram, W, H, INK, MUTED, PURPLE, TEAL

OUT = Path(__file__).resolve().parent


class ConnectedFramework(Diagram):
    def node(self, name, x, y, w, h, title, body, fill, stroke, body_size=25):
        self.rect(name, x, y, w, h, fill=fill, stroke=stroke, radius=15, width=2)
        subscript = 'L' if title.endswith('f_L') else None
        title_prefix = title[:-2] if subscript else title
        self.text(name+'_title',x+12,y+21,w-24,38,title_prefix,29,True,
                  parent=name,subscript=subscript)
        self.text(name+'_body',x+12,y+70,w-24,h-86,body,body_size,
                  color=MUTED,parent=name)

    def scene(self):
        self.text('title',60,25,1660,51,
                  'Context-aware Adaptive Temporal Selection',40,True,align='left')
        self.text('subtitle',62,83,1700,32,
                  'Proposed framework for battery SOH estimation',25,
                  color=MUTED,align='left')

        # One shared canvas: the secondary learning branch joins the Selector.
        self.node('supervision',380,180,390,138,'Lookback supervision',
                  ['OOF errors: LB10 / LB20 / LB50'],
                  '#F7F4FA','#A390B7',22)
        self.node('optimal_label',840,180,390,138,'Optimal lookback label',
                  ['Lowest seed-mean error'],
                  '#F7F4FA','#A390B7',24)
        self.text('train_note',380,140,850,31,
                  'Training batteries only · dashed arrows',23,
                  color=PURPLE,align='left')

        self.node('history',60,410,250,145,'Battery history',
                  ['V · I · dV · T'],
                  '#F3F5F7','#8A98A6',25)
        self.node('context',380,410,270,145,'Battery Context',
                  ['15 features'],
                  '#F0EBF7',PURPLE,27)
        self.node('selector',730,410,300,145,'Selector',
                  ['Choose L: 10 / 20 / 50'],
                  '#E8DEF4',PURPLE,24)
        self.node('window',1130,410,290,145,'Last L cycles',
                  ['L × 128 × 4'],
                  '#F0EBF7',PURPLE,28)
        self.node('predictor',1500,410,300,145,'Predictor f_L',
                  ['Selected SOH model'],
                  '#EAF5F2',TEAL,25)
        self.node('soh',1880,410,250,145,'SOH estimate',
                  ['ŷ'],
                  '#DDEFEA',TEAL,38)

        # Main connected left-to-right inference route.
        self.edge('history_to_context','history','context',[(310,482.5),(380,482.5)])
        self.edge('context_to_selector','context','selector',[(650,482.5),(730,482.5)])
        self.edge('selector_to_window','selector','window',[(1030,482.5),(1130,482.5)])
        self.edge('window_to_predictor','window','predictor',[(1420,482.5),(1500,482.5)])
        self.edge('predictor_to_soh','predictor','soh',[(1800,482.5),(1880,482.5)])

        # The window receives the original signals as well as the selected length.
        self.edge('raw_history_to_window','history','window',
                  [(185,555),(185,630),(1275,630),(1275,555)])
        self.text('signal_note',440,644,600,35,
                  'Past cycle signals',24,color=MUTED)

        # Dashed path is used to derive the selector's supervision, only in training.
        self.edge('training_history','history','supervision',
                  [(185,410),(185,249),(380,249)],True)
        self.edge('error_to_label','supervision','optimal_label',
                  [(770,249),(840,249)],True)
        self.edge('label_to_selector','optimal_label','selector',
                  [(1035,318),(1035,361),(880,361),(880,410)],True)

        self.node('evaluation',1655,660,475,125,'Accuracy & computational cost',
                  ['MAE / RMSE · mean L · total latency'],
                  '#F5F8F7','#8AAEA6',23)
        self.edge('soh_to_evaluation','soh','evaluation',[(2005,555),(2005,660)])
        self.text('evaluation_note',1655,795,475,28,
                  'Evaluate on unseen test batteries',21,color=MUTED)


diagram=ConnectedFramework()
mxfile=ET.Element('mxfile',{'host':'app.diagrams.net','type':'device','compressed':'false'})
page=ET.SubElement(mxfile,'diagram',{'id':'connected_framework','name':'Connected framework'})
page.append(diagram.model)
xml=ET.tostring(mxfile,encoding='unicode')
svg=diagram.svg(xml)
stem='soh_connected_framework'
(OUT/f'{stem}.drawio').write_text(xml,encoding='utf-8')
(OUT/f'{stem}.svg').write_text(svg,encoding='utf-8')
(OUT/f'{stem}.html').write_text(
    '<!doctype html><html><head><meta charset="utf-8"><style>'
    f'@page{{size:{W}px {H}px;margin:0}}'
    f'html,body{{margin:0;padding:0;width:{W}px;height:{H}px;overflow:hidden;background:white}}'
    'svg{display:block}*{-webkit-print-color-adjust:exact;print-color-adjust:exact}'
    '</style></head><body>'+svg+'</body></html>',encoding='utf-8')

cells=page.findall('./mxGraphModel/root/mxCell')
ids=[c.attrib['id'] for c in cells]
assert len(ids)==len(set(ids))
for cell in cells:
    for key in ('parent','source','target'):
        if key in cell.attrib:
            assert cell.attrib[key] in ids
print(f'{len(cells)} native cells; every connector has valid endpoints.')
print(OUT/f'{stem}.drawio')
