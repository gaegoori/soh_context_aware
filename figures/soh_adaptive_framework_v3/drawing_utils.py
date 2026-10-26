import xml.etree.ElementTree as ET
from html import escape
W, H = 2180, 850
INK = "#263442"
MUTED = "#596777"
PURPLE = "#755698"
DONE = "#F0EBF7"
TEAL = "#347D78"
PLAN = "#EAF5F2"
LINE = "#B3BEC9"

class Diagram:
    def __init__(self, progress=False):
            self.progress = progress
            self.model = ET.Element('mxGraphModel', {
                'dx': str(W), 'dy': str(H), 'grid': '1', 'gridSize': '10',
                'guides': '1', 'tooltips': '1', 'connect': '1', 'arrows': '1',
                'fold': '1', 'page': '1', 'pageScale': '1', 'pageWidth': str(W),
                'pageHeight': str(H), 'math': '0', 'shadow': '0',
            })
            self.root = ET.SubElement(self.model, 'root')
            ET.SubElement(self.root, 'mxCell', {'id': '0'})
            ET.SubElement(self.root, 'mxCell', {'id': '1', 'parent': '0'})
            self.coords = {'1': (0, 0, W, H)}
            self.rects, self.texts, self.edges = [], [], []
            self.ids = set(['0', '1'])
            self.scene()

    def vertex(self, name, x, y, w, h, style, value='', parent='1'):
            assert name not in self.ids, name
            self.ids.add(name)
            px, py, _, _ = self.coords[parent]
            node = ET.SubElement(self.root, 'mxCell', {
                'id': name, 'value': value, 'style': style,
                'vertex': '1', 'parent': parent,
            })
            ET.SubElement(node, 'mxGeometry', {
                'x': str(x-px), 'y': str(y-py),
                'width': str(w), 'height': str(h), 'as': 'geometry',
            })
            self.coords[name] = (x, y, w, h)

    def rect(self, name, x, y, w, h, fill='#FFFFFF', stroke=LINE,
                 radius=12, parent='1', width=1.5):
            style = (f'rounded=1;absoluteArcSize=1;arcSize={radius*2};whiteSpace=wrap;'
                     f'html=0;fillColor={fill};strokeColor={stroke};strokeWidth={width};'
                     'container=1;collapsible=0;recursiveResize=0;')
            self.vertex(name, x, y, w, h, style, parent=parent)
            self.rects.append(dict(x=x,y=y,w=w,h=h,fill=fill,stroke=stroke,r=radius,lw=width))
            return name

    def text(self, name, x, y, w, h, lines, size=23, bold=False,
                 color=INK, align='center', parent='1', subscript=None):
            if isinstance(lines, str):
                lines = lines.split('\n')
            style = (f'text;html={1 if subscript else 0};whiteSpace=wrap;overflow=hidden;rounded=0;'
                     f'strokeColor=none;fillColor=none;align={align};verticalAlign=middle;'
                     f'fontFamily=Arial;fontSize={size};fontColor={color};'
                     f'fontStyle={1 if bold else 0};spacing=0;resizable=1;')
            value = '\n'.join(lines)
            if subscript:
                value = escape(value) + '<sub style="font-size:72%;">' + escape(subscript) + '</sub>'
            self.vertex(name,x,y,w,h,style,value,parent)
            self.texts.append(dict(x=x,y=y,w=w,h=h,lines=lines,size=size,
                                   bold=bold,color=color,align=align,subscript=subscript))

    def box(self, name, x, y, w, h, title, lines, planned=False, size=23):
            self.rect(name,x,y,w,h,fill=PLAN if planned else DONE,
                      stroke=TEAL if planned else PURPLE)
            self.text(name+'_title',x+12,y+12,w-24,34,title,26,True,parent=name)
            self.text(name+'_body',x+14,y+52,w-28,h-63,lines,size,parent=name)
            return name

    def edge(self, name, source, target, points, dashed=False):
            assert source in self.ids and target in self.ids
            sx, sy, sw, sh = self.coords[source]
            tx, ty, tw, th = self.coords[target]
            ex, ey = (points[0][0]-sx)/sw, (points[0][1]-sy)/sh
            ix, iy = (points[-1][0]-tx)/tw, (points[-1][1]-ty)/th
            color = PURPLE if dashed else '#4C5867'
            style = (f'edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;'
                     f'jettySize=auto;html=0;endArrow=block;endFill=1;strokeWidth=2;'
                     f'strokeColor={color};dashed={1 if dashed else 0};'
                     f'exitX={ex};exitY={ey};exitDx=0;exitDy=0;exitPerimeter=0;'
                     f'entryX={ix};entryY={iy};entryDx=0;entryDy=0;entryPerimeter=0;')
            node = ET.SubElement(self.root,'mxCell',{
                'id':name,'value':'','style':style,'edge':'1','parent':'1',
                'source':source,'target':target,
            })
            self.ids.add(name)
            geo = ET.SubElement(node,'mxGeometry',{'relative':'1','as':'geometry'})
            if len(points)>2:
                arr = ET.SubElement(geo,'Array',{'as':'points'})
                for x,y in points[1:-1]:
                    ET.SubElement(arr,'mxPoint',{'x':str(x),'y':str(y)})
            self.edges.append(dict(points=points,dashed=dashed,color=color))

    def svg(self, mxfile):
            # The same native elements drive the SVG preview and the editable diagram.
            NS = 'http://www.w3.org/2000/svg'
            ET.register_namespace('',NS)
            def tag(t): return '{'+NS+'}'+t
            svg = ET.Element(tag('svg'), {'width':str(W),'height':str(H),
                'viewBox':f'0 0 {W} {H}', 'content':mxfile, 'role':'img',
                'aria-label':'Context-aware adaptive temporal selection for battery SOH'})
            defs=ET.SubElement(svg,tag('defs'))
            for name,color in [('arrow','#4C5867'),('control',PURPLE)]:
                m=ET.SubElement(defs,tag('marker'),{'id':name,'viewBox':'0 0 10 10',
                    'refX':'9','refY':'5','markerWidth':'8','markerHeight':'8',
                    'orient':'auto-start-reverse','markerUnits':'userSpaceOnUse'})
                ET.SubElement(m,tag('path'),{'d':'M 0 0 L 10 5 L 0 10 z','fill':color})
            ET.SubElement(svg,tag('rect'),{'width':str(W),'height':str(H),'fill':'#FFFFFF'})
            for r in self.rects:
                ET.SubElement(svg,tag('rect'),{k:str(v) for k,v in {
                    'x':r['x'],'y':r['y'],'width':r['w'],'height':r['h'],
                    'rx':r['r'],'fill':r['fill'],'stroke':r['stroke'],'stroke-width':r['lw']}.items()})
            for e in self.edges:
                a={'points':' '.join(f'{x},{y}' for x,y in e['points']),
                    'fill':'none','stroke':e['color'],'stroke-width':'2.1',
                    'stroke-linejoin':'round','marker-end':f'url(#{"control" if e["dashed"] else "arrow"})'}
                if e['dashed']: a['stroke-dasharray']='9 6'
                ET.SubElement(svg,tag('polyline'),a)
            for t in self.texts:
                x=t['x'] if t['align']=='left' else t['x']+t['w']/2
                gap=t['size']*1.28
                y=t['y']+t['h']/2-(len(t['lines'])-1)*gap/2+t['size']*0.35
                node=ET.SubElement(svg,tag('text'),{'x':str(x),'y':str(y),
                    'font-family':'Arial, Helvetica, sans-serif','font-size':str(t['size']),
                    'font-weight':'700' if t['bold'] else '400','fill':t['color'],
                    'text-anchor':'start' if t['align']=='left' else 'middle'})
                for i,line in enumerate(t['lines']):
                    span = ET.SubElement(node,tag('tspan'),{'x':str(x),'y':str(y+i*gap)})
                    span.text=line
                    if t.get('subscript'):
                        ET.SubElement(span,tag('tspan'),{
                            'baseline-shift':'sub','font-size':'72%',
                        }).text=t['subscript']
            return ET.tostring(svg,encoding='unicode')
