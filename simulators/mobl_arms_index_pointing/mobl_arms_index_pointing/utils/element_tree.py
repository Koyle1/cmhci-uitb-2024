import xml.etree.ElementTree as ET

def create(root, name):
  # Creates an element if it doesn't exist
  element = root.find(name)
  if element is None:
    root.append(ET.Element(name))

def copy_or_append(name, src, dst):
    #Hängt den Inhalt eines XML Files an ein anderes dran
    #Wird verwendet um bm_model in die Task simulation zu kopieren
  element = src.find(name)

  if dst.find(name) is None:
    dst.append(element)
  else:
    dst.find(name).append(element)

def copy_children(name, src, dst, exclude=None):

  # Check if there is something to copy
  elements = src.find(name)

  if elements is not None:

    # Create an element if necessary
    create(dst, name)

    # Copy each element except ones that are excluded
    for element in elements:
        #Überspringe wenn exclude Kriterien erfüllt, meistens bei Skybox
      if exclude is not None and \
          element.tag == exclude["tag"] and element.attrib[exclude["attrib"]] == exclude["name"]:
        continue
      else:
          #Hänge das Kindelement an die Wurzel in der neuen Datei an
        dst.find(name).append(element)